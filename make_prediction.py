#! /usr/bin/env python


import numpy as np
import torch
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import sys
import importlib

from load_dataset import WildfireDataset
from predict import predict_with_weights_for_models, final_ensemble_predictions, save_rdata, save_obs_to_rdata

from fastai.basics import DataLoader



def make_dataloader(fpath="padded_normalized.npz", device="cpu"):
    numpy_file = np.load(fpath)
    x, master_mask, BA_mask, CNT_mask, years, months = [
        torch.from_numpy(numpy_file[s]).to(torch.device(device))
        for s in [
            "data",
            "master_mask",
            "BA_mask",
            "CNT_mask",
            "years",
            "months",
        ]
    ]
    
    master_mask = master_mask.to(bool)
    BA_mask = BA_mask.to(bool)
    CNT_mask = CNT_mask.to(bool)

    continuous_time = years + months / 12
    continuous_time = (continuous_time - continuous_time.mean()) / continuous_time.std()

    mask = torch.stack((CNT_mask, BA_mask), dim=1)

    all_ds = WildfireDataset(
        x.float(),
        years.long(),
        months.long(),
        continuous_time.float(),
        x[:, :2].float(),
        mask.float(),
    )

    all_dl = DataLoader(all_ds, shuffle=False, batch_size=1, drop_last=False, device=torch.device(device))
    
    return all_dl

def create_model_name(model_data,
                      distribution_type="ZeroModifiedLogNormal",
                      latent_size=64,
                      default_K=768,
                      batch_size=2,
                      kernel_size=5,
                      dropout=0.1,
                      run_no=None
                     ):
    assert run_no is not None
    assert model_data['run'] == run_no  # BUGCHECK! - check if run_no from create_model (passed argument run_no) equals run_no from parsing output file name (passed in model_data dict)
    
    return Path("models") / f"CMIWAE-run{model_data['run']}-{distribution_type}-ls{latent_size}-K{default_K}-bs{batch_size}-seed{model_data['seed']}.pth"
   
def make_predictions_from_data_improved(model,
                                       models_per_year,  # dict - list of models to use per every test year
                                       test_years,
                                       test_dl,
                                       CNT_unknown_idx,  # list of tensors of idxs per month
                                       BA_unknown_idx,  # list of tensors of idxs per month
                                       N=1,
                                       default_K=128,
                                       device=None,
                                       rdata_filename="prediction_BlackBox.RData",
                                      ):
    """
    Improved version of function make_predictions_from_data.
    Works per month (not year) bases, and does not write to disk.
    """

    test_idxs = sorted([y * 7 + m for m in range(7) for y in test_years])
    
    all_idxs_prediction_CNT = []
    all_idxs_prediction_BA = []
    for month_idx in tqdm(test_idxs):
        assert CNT_unknown_idx[month_idx] is not None
        assert BA_unknown_idx[month_idx] is not None

        year_idx = month_idx // 7
        
        print(f"Predicting for month {month_idx}, {len(models_per_year[year_idx])} models...")
        models_lws, models_CNT_cdfs, models_BA_cdfs = predict_with_weights_for_models(
                                                        model,
                                                        models_per_year[year_idx],
                                                        test_dl,
                                                        CNT_unknown_idx[month_idx],
                                                        BA_unknown_idx[month_idx],
                                                        N=N,
                                                        default_K=default_K,
                                                        device=device,
                                                        save_to_disk=False,
                                                    )

        print(f"Ensembling predictions for month {month_idx}...")
        prediction_CNT, prediction_BA = final_ensemble_predictions(models_per_year[year_idx],
                                                                   load_from_disk=False,
                                                                   all_lws_per_model=models_lws,
                                                                   all_CNT_cdfs_per_model=models_CNT_cdfs,
                                                                   all_BA_cdfs_per_model=models_BA_cdfs
                                                                  )
        del models_lws, models_CNT_cdfs, models_BA_cdfs
        all_idxs_prediction_CNT.append(prediction_CNT)
        all_idxs_prediction_BA.append(prediction_BA)

    final_prediction_CNT = np.concatenate(all_idxs_prediction_CNT)
    final_prediction_BA = np.concatenate(all_idxs_prediction_BA)

    print("Saving final prediction to .RData")
    print(f"final_prediction_CNT shape: {final_prediction_CNT.shape}")
    print(f"final_prediction_BA shape: {final_prediction_BA.shape}")
    save_rdata(final_prediction_CNT, final_prediction_BA, rdata_filename=rdata_filename)

def load_CNT_and_BA_unknown_idx(fpath="padded_normalized.npz",
                                device="cuda",
                                load_data_per_year=True,  # if false, prepares list of indxs tensors per month bases
                               ):
    """
    Loads and formats CNT_unknown_idx and BA_unknown_idx from data file, used
    for generating competition prediction.
    """
    devider = 7 if load_data_per_year else 1
    
    numpy_file = np.load(fpath)

    CNT_unknown_idx = torch.from_numpy(numpy_file['CNT_unknown_idx']).to(torch.device(device)).T
    CNT_unknown_idx_l = []
    for k, g in tqdm(groupby(CNT_unknown_idx, key=lambda i: i[0] // devider), total=77 // devider):
        idxs = list(g)
        ts = []
        for idx in idxs:
            ts.append(torch.unsqueeze(idx, 0))          
        CNT_unknown_idx_l.append(torch.cat(ts).T.cpu().numpy())

    BA_unknown_idx = torch.from_numpy(numpy_file['BA_unknown_idx']).to(torch.device(device)).T
    BA_unknown_idx_l = []
    for k, g in tqdm(groupby(BA_unknown_idx, key=lambda i: i[0] // devider), total=77 // devider):
        idxs = list(g)
        ts = []
        for idx in idxs:
            ts.append(torch.unsqueeze(idx, 0))   
        BA_unknown_idx_l.append(torch.cat(ts).T.cpu().numpy())
    
    CNT_unknown_idx_l_out, BA_unknown_idx_l_out = [], []
    if load_data_per_year:
        idx = 0
        for y in range(0,23):
            if y % 2 == 0:
                CNT_unknown_idx_l_out.append(None)
                BA_unknown_idx_l_out.append(None)
            else:
                CNT_unknown_idx_l_out.append(CNT_unknown_idx_l[idx])
                BA_unknown_idx_l_out.append(BA_unknown_idx_l[idx])
                idx += 1
        assert idx == 11
    else:
        idx = 0
        for m_idx in range(0,161):
            y = m_idx // 7
            if y % 2 == 0:
                CNT_unknown_idx_l_out.append(None)
                BA_unknown_idx_l_out.append(None)
            else:
                CNT_unknown_idx_l_out.append(CNT_unknown_idx_l[idx])
                BA_unknown_idx_l_out.append(BA_unknown_idx_l[idx])
                idx += 1
        assert idx == 77
    
    return CNT_unknown_idx_l_out, BA_unknown_idx_l_out

def save_observation_data(CNT_unknown_idx, BA_unknown_idx, rdata_filename="observation.RData", fpath = "padded_normalized_full.npz"):
    # idxs should be list of idxs for every year!
    assert len(CNT_unknown_idx) == 23
    assert len(BA_unknown_idx) == 23
    CNT_unknown_idx_test_all = np.concatenate(CNT_unknown_idx[1:23:2], axis=1)
    BA_unknown_idx_test_all = np.concatenate(BA_unknown_idx[1:23:2], axis=1)

    numpy_file = np.load(fpath)

    CNT_data = numpy_file['data'][:, 0]
    BA_data = numpy_file['data'][:, 1]

    CNT_unknown_idx_test_all_l = CNT_unknown_idx_test_all.T.tolist()
    CNT_data_flat = []
    for elem in CNT_unknown_idx_test_all_l:
        t, h, w = elem
        CNT_data_flat.append(CNT_data[t, h, w])
    CNT_data_flat_np = np.array(CNT_data_flat)

    BA_unknown_idx_test_all_l = BA_unknown_idx_test_all.T.tolist()
    BA_data_flat = []
    for elem in BA_unknown_idx_test_all_l:
        t, h, w = elem
        BA_data_flat.append(BA_data[t, h, w])
    BA_data_flat_np = np.array(BA_data_flat)

    print(f"Saving observations to {rdata_filename}")
    save_obs_to_rdata(CNT_data_flat, BA_data_flat, rdata_filename=rdata_filename)
    
def load_params_from_command_line():
    ###########################################################################
    #  P A R A M E T E R S
    ###########################################################################   
    if len(sys.argv) == 6:
        run = int(sys.argv[1])
        fpath = f"outputs/output-run{run}-idx*.txt"
        prediction_no = int(sys.argv[2])
        device = sys.argv[3]  # for prediction using only one GPU, device="cpu" will use CPU only - very slow!
        N = int(sys.argv[4])
        K_sampling = int(sys.argv[5])
    else:
        raise RuntimeError("Please specify command line arguments: run_no pred_no cuda_device N K_sampling")
    ###########################################################################
    
    return run, fpath, prediction_no, device, N, K_sampling

def create_model_for_run(run, device):
    train_run = importlib.import_module(f"train_run_{run}")
    model, model_params = train_run.create_model(train_run.iwae_sampling_params()['K'], device=device)
    model_params["batch_size"] = train_run.dataloader_params()['batch_size']
    
    return model, model_params