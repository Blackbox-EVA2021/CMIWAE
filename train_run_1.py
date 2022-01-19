#! /usr/bin/env python

# launch using:
# CUDA_VISIBLE_DEVICES=2,3,4,5 python -m fastai.launch scriptname.py

# 1. final version

import sys
from functools import lru_cache
import time
import parse
from pathlib import Path

import torch
from torch.functional import split

from fastai.distributed import *
from fastai.callback.all import *
from fastai.torch_core import rank_distrib
from fastai.basics import *

from models import (
    Enc,
    Dec,
    Enc_cond,
    Dec_cond,
    MinimalBlock,
    MinimalBlockTranspose,
    IllegalArgument,
)
from load_dataset import load_data_deterministic
from dreg_loss import dreg_loss
from cmiwae import CMIWAE
from distributions import ZeroModifiedLogNormal, Binned
from callbacks import TestScoreCallback, MySaveModelCallback


beg_time = time.perf_counter()

EPS = 1e-6


def build_ZeroModifiedLogNormal_output(params, eps=EPS):
    # params.shape == [K, bs, num_params, h, w]
    assert len(params.shape) == 5
    assert params.shape[-3] == 5

    # prepare parameters for px_z
    # logit has only one channel,
    # so the zero-or-not Bernoulli event is common for BA and CNT channels
    # this guarantees that BA and CNT are both zero at the same time
    logit = params[..., 0:1, :, :]

    positive_params = torch.nn.functional.softplus(params[..., 1:, :, :]) + eps
    meanlog, sdlog = positive_params.split(2, dim=-3)

    return ZeroModifiedLogNormal(logit, meanlog, sdlog)

def build_Binned_output(params, eps=EPS):
    # params.shape == [K, bs, num_params, h, w]
    assert len(params.shape) == 5
    assert params.shape[-3] == 57

    return Binned(params)


distribution_info = {"ZeroModifiedLogNormal": (5, build_ZeroModifiedLogNormal_output),
                     "Binned": (57, build_Binned_output)
                    }


class WildfireCMIWAE(CMIWAE):
    def __init__(
        self,
        latent_size=128,
        enc=None,
        enc_cond=None,
        dec_cond=None,
        embedding_dim=2,
        default_K=1,
        h=64,
        w=128,
        eps=1e-6,
        distribution_type="ZeroModifiedLogNormal",
    ):
        super().__init__(latent_size=latent_size, default_K=default_K)
        self.eps = eps

        # parameters for the approximate posterior qz_x
        self.enc = enc
        self._enc_cond = enc_cond
        self.enc_cond = lru_cache(maxsize=1)(self.enc_cond)

        # parameters for px_z
        self.dec_cond = dec_cond

        self.month_embedding = torch.nn.Embedding(
            num_embeddings=7, embedding_dim=embedding_dim
        )

        self.distribution_type = distribution_type
        _, self.build_output_distribution = distribution_info[distribution_type]

    def enc_cond(self, *args, **kwargs):
        return self._enc_cond(*args, **kwargs)

    def prepare_input(self, *inp):
        x_all, year, month, continuous_time = inp
        bs, _, h, w = x_all.shape

        continuous_time = continuous_time.view(bs, 1, 1, 1).expand(bs, 1, h, w)

        month_embedded = (
            self.month_embedding(month - 3).view(bs, -1,
                                                 1, 1).expand(bs, -1, h, w)
        )  # .shape == [bs, embedding, h, w]

        c = torch.cat(
            (x_all[:, 2:], continuous_time, month_embedded),
            dim=1,
        )
        almost_log = torch.log(1 + x_all[:, :2])
        x = torch.cat((x_all[:, :2], almost_log), dim=1)

        return x, c

    def qz_xc(self, x, c):
        xc = torch.cat((x, c), dim=1)
        # enc_params.shape == [bs, num_params, latent_size]
        qz_xc_params = self.enc(xc)
        
        # iterate over num_params
        loc, scale = torch.movedim(qz_xc_params, 1, 0)
        scale = (
            torch.nn.functional.softplus(scale) + self.eps
        )  # positive scale, arbitrarily wide or narrow

        return torch.distributions.Normal(loc, scale)

    def pz_c(self, c):
        # pz_c_params.shape == [bs, num_params, latent_size]
        _, pz_c_params = self.enc_cond(c)
        loc, scale = torch.movedim(pz_c_params, 1, 0)
        scale = (
            torch.nn.functional.softplus(scale) + self.eps
        )  # positive scale, arbitrarily wide or narrow

        return torch.distributions.Normal(loc, scale)

    def px_zc(self, z, c):
        K, bs, zdim = z.shape
        assert zdim == self.latent_size

        # Conv2D expects the shape to be [bs, ch, h, w]
        # put K MC samples into batch dimension and after calculation reshape back
        # the second output is intended for pz_c only
        cond_out, _ = self.enc_cond(c)
        cond_out = [l.repeat(K, 1, 1, 1) for l in cond_out]

        px_zc_params = self.dec_cond(
            z.view(-1, zdim), cond_out  # [K * bs, zdim]
        )  # px_z_params.shape == [K * bs, num_params, h, w]
        px_zc_params = px_zc_params.view(
            K, bs, *px_zc_params.shape[1:]
        )  # px_zc_params.shape == [K, bs, num_params, h, w]

        return self.build_output_distribution(px_zc_params, self.eps)


def create_model(K, device="cuda", *args, **kwargs):
    ###########################################################################
    # M O D E L    P A R A M E T E R S
    ###########################################################################
    latent_size = 64
    embedding_dim = 3

    distribution_type = "ZeroModifiedLogNormal"
    #distribution_type = "Binned"

    kernel_size = 5
    use_BatchNorm = True
    init_params = False
    dropout = 0.1
    nonlin = torch.nn.Softplus()
    ###########################################################################

    # Create encoder-decoder models

    enc_in_ch = 35 + 1 * embedding_dim + 3
    dec_out_ch, _ = distribution_info[distribution_type]

    # Encoder
    enc = Enc(enc_in_ch,
              MinimalBlock,
              latent_size=latent_size,
              kernel_size=kernel_size,
              dropout=dropout,
              layers_num=[1, 1, 1, 1, 1, 1],
              channels=[64, 64, 64, 64, 64, 64],
              use_BatchNorm=use_BatchNorm,
              init_params=init_params,
              nonlin=nonlin,
              )

    # Conditional encoder
    enc_cond = Enc_cond(enc_in_ch - 4,
                        MinimalBlock,
                        latent_size=latent_size,
                        kernel_size=kernel_size,
                        dropout=dropout,
                        layers_num=[1] * 6,
                        channels=[32, 32, 32, 32, 32, 32],
                        use_BatchNorm=use_BatchNorm,
                        init_params=init_params,
                        nonlin=nonlin,
                        )
    # Conditional decoder
    dec_cond = Dec_cond(dec_out_ch,
                        MinimalBlockTranspose,
                        latent_size=latent_size,
                        kernel_size=kernel_size,
                        dropout=dropout,
                        layers_num=[1] * 6,
                        channels=[8, 8, 16, 16, 32, 32],
                        cond_in_ch=[8, 8, 16, 16, 32, 32],
                        use_BatchNorm=use_BatchNorm,
                        init_params=init_params,
                        nonlin=nonlin,
                        )

    model = WildfireCMIWAE(
        enc=enc,
        enc_cond=enc_cond,
        dec_cond=dec_cond,
        latent_size=latent_size,
        embedding_dim=embedding_dim,
        default_K=K,
        distribution_type=distribution_type,
    ).to(device)

    # Print number of parameters data
    if not rank_distrib():
        print(f"Enc num param: {enc.count_num_of_parameters()}")
        print(f"Enc fc input size: {enc.get_fc_input_size()}")
        print(f"Enc cond num param: {enc_cond.count_num_of_parameters()}")
        print(f"Dec cond num param: {dec_cond.count_num_of_parameters()}")
        
    # Get run_no form script file name
    res = parse.parse("train_run_{run_no:d}.py", Path(__file__).name)
    assert res is not None
    
    return model, {"distribution_type" : distribution_type,
                   "latent_size" : latent_size,
                   "default_K" : K,
                   "kernel_size" : kernel_size,
                   "dropout" : dropout,
                   "run_no" : res.named['run_no']  # returns run_no encoded in script file name - bugcheck! - must match expected run_no when making predictions
                  }

def dataloader_params():
    ###########################################################################
    #  D A T A L O A D E R    P A R A M E T E R S
    ###########################################################################
    return({'batch_size' : 2,
           })
    ###########################################################################
    
def iwae_sampling_params():
    ###########################################################################
    #  I W A E    S A M P L I N G    P A R A M E T E R S
    ###########################################################################
    return({#'K' : 300,  # binned distribution works
            'K' : 768,
           })
    ###########################################################################
             

if __name__ == "__main__":
    setup_distrib(rank_distrib())

    # Parse command line args

    valid_set_idx = int(sys.argv[1])
    RUN_NO = int(sys.argv[2])
    if not rank_distrib():
        print(f"Run number: {RUN_NO}")
        print(f"Using validation set index: {valid_set_idx}")
        
    batch_size = dataloader_params()['batch_size']

    # Load dataset
    data, mean, std, master_mask, valid_years = load_data_deterministic(
        "padded_normalized.npz",
        batch_size=batch_size,
        valid_set_idx=valid_set_idx,
        valid_size=2,
    )
    
    # Create model
    model, _ = create_model(iwae_sampling_params()['K'])

    # Create Learner object
    learner = Learner(data, model, dreg_loss, Adam)

    my_callback = TestScoreCallback(device=data.device)
    learner.metrics += ValueMetric(my_callback.CNT_score, 'CNT_score')
    learner.metrics += ValueMetric(my_callback.BA_score, 'BA_score')
    learner.metrics += ValueMetric(my_callback.total_score, 'total_score')

    ###########################################################################
    #  T R A I N    P A R A M E T E R S
    ###########################################################################
    epochs = 100
    lr_max = 3e-3
    ########################################################################### 
    
    with learner.distrib_ctx():
        callbacks = [my_callback]
        if not rank_distrib():
            callbacks.append(
                MySaveModelCallback(
                    monitor='total_score',
                    comp=np.less,
                    fname=f"CMIWAE-run{RUN_NO}-{model.distribution_type}-ls{model.latent_size}-K{model.default_K}-bs{batch_size}-seed{valid_set_idx}",
                )
            )       
        
        # TRAIN LOOP
        learner.fit_one_cycle(epochs, lr_max=lr_max, div_final=1e2, cbs=callbacks)

    # After train output
    if not rank_distrib():
        print((
            f"Minimum score: {my_callback.min_score.cpu().numpy()}, "
            f"sum: {my_callback.min_score.sum()}"
        ))
        print(f"Validation set years: {valid_years}")
        print(f"Total time: {(time.perf_counter() - beg_time):.2f}s")
