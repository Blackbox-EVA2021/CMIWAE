#! /usr/bin/env python

from pathlib import Path
from itertools import groupby

import numpy as np
import torch

from tqdm import tqdm

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

base = importr("base")
rpy2.robjects.numpy2ri.activate()


def predict_with_weights(
    model, dl, CNT_unknown_idx, BA_unknown_idx, N=256, default_K=512, device=None,
):
    if device is None:
        device = dl.device

    model.eval()

    model.default_K = default_K

    CNT_grouped = groupby(CNT_unknown_idx.T, key=lambda i: i[0])
    BA_grouped = groupby(BA_unknown_idx.T, key=lambda i: i[0])

    CNT_bins = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18,
                             20, 22, 24, 26, 28, 30, 40, 50, 60, 70, 80, 90, 100])
    BA_bins = torch.tensor([0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                            150, 200, 250, 300, 400, 500, 1000, 1500, 2000,
                            5000, 10000, 20000, 30000, 40000, 50000, 100000])

    # 28x2 bins reshaped to broadcast with px_z [K, bs, 2, h, w]
    bins = (
        torch.stack((CNT_bins, BA_bins), dim=-1)
        .view(28, 1, 1, 2, 1, 1)
        .float()
        .to(device)
    )

    dl.bs = 1 # very important - DO NOT CHANGE!
    batch_it = enumerate(dl)

    all_lws, all_CNT_cdfs, all_BA_cdfs = {}, {}, {}
    for (t_idx, CNT_idx), (_, BA_idx) in zip(CNT_grouped, BA_grouped):
        while True:
            t_idx_data, (*x, (x_target, mask, *other_masks)) = next(batch_it)
            if t_idx == t_idx_data:
                break

        # if BA is 0, assume CNT is 0 and vice versa
        known_zeros = (mask != 0) & (x[0][:, :2] == 0)
        CNT_known_zeros = known_zeros[0, 0]
        BA_known_zeros = known_zeros[0, 1]

        # if BA is not 0, assume CNT is not 0 and vice versa
        known_non_zeros = (mask != 0) & (x[0][:, :2] != 0)
        CNT_known_non_zeros = known_non_zeros[0, 0]
        BA_known_non_zeros = known_non_zeros[0, 1]

        CNT_idx = tuple(zip(*CNT_idx))[1:]
        BA_idx = tuple(zip(*BA_idx))[1:]

        lws, CNT_cdfs, BA_cdfs = [], [], []
        for _ in tqdm(range(N), leave=False):
            with torch.no_grad():
                qz_x, px_z, pz, z = model(*x)

            single_channel_mask = mask.bool().any(dim=-3, keepdim=True) # [bs, 1, h, w]
            lqz_x = qz_x.log_prob(z).sum(-1)
            lpx_z = (
                px_z.log_prob(x_target)
                .mul(single_channel_mask)
                .view(*px_z.batch_shape[:-3], -1)
                .sum(-1)
            )  # px_z.shape == [K, bs, 2, w, h], lpx_z.shape == [K, bs]
            lpz = pz.log_prob(z).sum(-1)
            lw = lpz + lpx_z - lqz_x  # lw.shape == [K, bs]

            # fix known zeros
            px_z.logit[..., CNT_known_zeros] = 1e6
            px_z.logit[..., BA_known_zeros] = 1e6

            # fix known non-zeros
            px_z.logit[..., CNT_known_non_zeros] = -1e6
            px_z.logit[..., BA_known_non_zeros] = -1e6

            cdf = (
                px_z.cdf(bins).detach().cpu()
            )  # shape = [28(bins), K, 1(bs), 2(CNT+BA), h, w]

            # fix known zeros
            # where BA is zero, CNT is zero as well, so CNT CDF is 1 for every bin, K, bs index
            #cdf[:, :, :, 0, BA_known_zeros] = 1
            # where CNT is zero, BA is zero as well, so BA CDF is 1 for every bin, K, bs index
            #cdf[:, :, :, 1, CNT_known_zeros] = 1

            CNT_cdfs.append(
                cdf[:, :, 0, 0, CNT_idx[0], CNT_idx[1]].detach().cpu()
            )  # list of shapes [28, K, missing data points]
            BA_cdfs.append(
                cdf[:, :, 0, 1, BA_idx[0], BA_idx[1]].detach().cpu()
            )  # list of shapes [28, K, missing data points]
            lws.append(
                lw[:, 0].unsqueeze(0).unsqueeze(-1).detach().cpu()
            )  # list of shapes [1, K, 1]

        all_lws[t_idx] = torch.cat(lws, dim=1).numpy()  # [1, N*K, 1]
        all_CNT_cdfs[t_idx] = torch.cat(
            CNT_cdfs, dim=1
        ).numpy()  # [28, N*K, missing data points]
        all_BA_cdfs[t_idx] = torch.cat(
            BA_cdfs, dim=1
        ).numpy()  # [28, N*K, missing data points]

    return all_lws, all_CNT_cdfs, all_BA_cdfs


def stringify_keys(d):
    return {str(k): v for k, v in d.items()}


def intify_keys(d):
    return {int(k): v for k, v in d.items()}


def make_new_filename(path, append_stem, suffix, directory="temp"):
    new_name = path.stem + append_stem + suffix
    return Path(directory) / Path(new_name)


def predict_with_weights_for_models(
    model,
    model_filenames,
    dl,
    CNT_unknown_idx,
    BA_unknown_idx,
    N=256,
    default_K=512,
    device=None,
    save_to_disk=True
):
    models_lws, models_CNT_cdfs, models_BA_cdfs = [], [], []
    for filename in tqdm(model_filenames):
        model.load_state_dict(torch.load(filename, map_location=device))
        all_lws, all_CNT_cdfs, all_BA_cdfs = predict_with_weights(
            model, dl, CNT_unknown_idx, BA_unknown_idx, N, default_K, device,
        )

        if save_to_disk:
            np.savez(make_new_filename(filename, "_lws", ".npz"),
                     **stringify_keys(all_lws))
            np.savez(
                make_new_filename(filename, "_CNT_cdfs", ".npz"),
                **stringify_keys(all_CNT_cdfs)
            )
            np.savez(
                make_new_filename(filename, "_BA_cdfs", ".npz"),
                **stringify_keys(all_BA_cdfs)
            )
        else:
            models_lws.append(all_lws)
            models_CNT_cdfs.append(all_CNT_cdfs)
            models_BA_cdfs.append(all_BA_cdfs)
    
    if not save_to_disk:
        return models_lws, models_CNT_cdfs, models_BA_cdfs
            


def final_ensemble_predictions(model_filenames,
                               load_from_disk=True,
                               all_lws_per_model=None,
                               all_CNT_cdfs_per_model=None,
                               all_BA_cdfs_per_model=None,
                              ):
    if load_from_disk:
        all_lws_per_model = [
            intify_keys(np.load(make_new_filename(filename, "_lws", ".npz")))
            for filename in tqdm(model_filenames, leave=False)
        ]
        all_CNT_cdfs_per_model = [
            intify_keys(np.load(make_new_filename(filename, "_CNT_cdfs", ".npz")))
            for filename in tqdm(model_filenames, leave=False)
        ]
        all_BA_cdfs_per_model = [
            intify_keys(np.load(make_new_filename(filename, "_BA_cdfs", ".npz")))
            for filename in tqdm(model_filenames, leave=False)
        ]
    else:
        assert all_lws_per_model is not None
        assert all_CNT_cdfs_per_model is not None
        assert all_BA_cdfs_per_model is not None

    # get month indices
    for l in all_lws_per_model:
        months = sorted(l.keys())
        break  # should be the same for all models

    prediction_CNT, prediction_BA = [], []
    for t_idx in tqdm(months, leave=False):
        lws = torch.cat(
            [torch.from_numpy(l[t_idx]) for l in all_lws_per_model], dim=1
        )  # [1, total preds, 1]

        ws = torch.nn.functional.softmax(lws, dim=1).numpy()
        del lws

        CNT_cdfs = np.concatenate(
            [l[t_idx] for l in all_CNT_cdfs_per_model], axis=1
        )  # [28, total preds, missing data points]
        prediction_CNT.append(
            (ws * CNT_cdfs).sum(axis=1).T
        )  # list of arrays of shape [missing data points, 28]
        del CNT_cdfs
    
        BA_cdfs = np.concatenate(
            [l[t_idx] for l in all_BA_cdfs_per_model], axis=1
        )  # [28, total preds, missing data points]
        prediction_BA.append(
            (ws * BA_cdfs).sum(axis=1).T
        )  # list of arrays of shape [missing data points, 28]
        del BA_cdfs

    prediction_CNT, prediction_BA = [
        np.concatenate(a, axis=0) for a in [prediction_CNT, prediction_BA]
    ]

    return prediction_CNT, prediction_BA


def save_rdata(
    prediction_CNT, prediction_BA, rdata_filename="prediction_BlackBox.RData"
):
    def assign_to_rmat(name, mat):
        nr, nc = mat.shape
        rmat = ro.r.matrix(mat, nrow=nr, ncol=nc)

        ro.r.assign(name, rmat)

    assign_to_rmat("prediction_cnt", prediction_CNT)
    assign_to_rmat("prediction_ba", prediction_BA)

    base.save("prediction_cnt", "prediction_ba", file=rdata_filename)


def save_obs_to_rdata(
    obs_CNT, obs_BA, rdata_filename="mock_obs_BlackBox.RData"
):
    def assign_to_rmat(name, vect):
        rvect = ro.vectors.FloatVector(vect)

        ro.r.assign(name, rvect)

    assign_to_rmat("obs_cnt", obs_CNT)
    assign_to_rmat("obs_ba", obs_BA)

    base.save("obs_cnt", "obs_ba", file=rdata_filename)
