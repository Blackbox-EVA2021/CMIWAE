#! /usr/bin/env python

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset
from fastai.basics import DataLoader
from fastai.basics import DataLoaders

from numpy.random import default_rng
from itertools import combinations


class WildfireDataset(TensorDataset):
    def __init__(
        self,
        x,
        years,
        months,
        continuous_time,
        target,
        *masks,
        **kwargs
    ):
        super().__init__(x, years, months, continuous_time, target, *masks, **kwargs)

    def __getitem__(self, idx):
        x, years, months, continuous_time, target, *masks = super().__getitem__(idx)
        
        return (
            x,
            years,
            months,
            continuous_time,
            (target, *masks),
        )  # the last element is unpacked as target for the loss fn



def load_data_deterministic(path,
                            batch_size=32,
                            device="cuda",
                            valid_set_idx=0,
                            valid_size=2
                           ):
    
    numpy_file = np.load(path)
    x, master_mask, BA_mask, CNT_mask, years, months, mean, std = [
        torch.from_numpy(numpy_file[s]).to(torch.device(device))
        for s in [
            "data",
            "master_mask",
            "BA_mask",
            "CNT_mask",
            "years",
            "months",
            "mean",
            "std",
        ]
    ]
    BA_mask = BA_mask.to(bool)
    CNT_mask = CNT_mask.to(bool)

    mask = torch.stack((CNT_mask, BA_mask), dim=1)

    # calculate normalized continuous time from year and month
    continuous_time = years + months / 12
    continuous_time = (continuous_time - continuous_time.mean()) / continuous_time.std()

  
    ############################################################
    # Creation of train, valid, (test) dataset split
    ############################################################
    
    all_idx = list(range(x.shape[0]))
    assert x.shape[0] == 161
        
    
    # FIRST. Create train and valid datasets for loss evaluation
    
    all_ds = WildfireDataset(
        x.float(),
        years.long(),
        months.long(),
        continuous_time.float(),
        x[:, :2].float(),
        mask.float(),
    )
    
    assert isinstance(valid_size, int)  # valid size is number of whole years to use
    assert valid_size >= 1

    full_years=list(range(0, 23, 2))  # All years indices, having full data - HARDCODED
    assert len(full_years) == 12  # There should be 12 full years of trivial masks - HARDCODED
    
    all_years=list(range(0, 23))
    valid_years=list(list(combinations(all_years, valid_size))[valid_set_idx])
    valid_idx = sorted([y * 7 + m for m in range(7) for y in valid_years])

    # train set is all month indices that are not in valid set
    train_idx = sorted(list(set(all_idx) - set(valid_idx)))

    mask_idx = [idx for idx in range(x.shape[0]) if (idx // 7) % 2 == 1]  # Indeces of all nontrivial masks
    assert len(mask_idx) == 7 * 11  # There should be 11 years of this masks - HARDCODED
    mask_idx = sorted(list(set(mask_idx) - set(valid_idx)))  # remove all masks that are allready used in validation set

    rng1 = default_rng(seed=valid_set_idx) # rng1 initialised by seed valid_set_idx
    rng1.shuffle(mask_idx)  # Order of mask indices is randomised

    train_ds = Subset(all_ds, train_idx)
    valid_ds = Subset(all_ds, valid_idx)    

    # SECOND. Create reconstruct_valid_ds for validation mock score calculation
    
    extra_dmg_mask = torch.zeros(mask.shape, dtype=mask.dtype, device=torch.device(device))  # create (at first) blank extra damage mask of mask shape
    extra_dmg_mask[valid_idx] = mask[mask_idx[:len(valid_idx)]]  # use randomised nontrivial masks to apply damage in valid years
    
    dmg_x = torch.cat((x[:, :2] * extra_dmg_mask, x[:, 2:]), dim=1)  # create extra damadged data
    dmg_mask = mask & extra_dmg_mask  # make additional subtraction on every mask, keeping only the elements in the intersection of mask and extra_dmg_mask
    new_damage_mask = mask & ~extra_dmg_mask  # the inverse of dmg_mask contained inside mask, equal to mask & ~dmg_mask, use DeMorgan
    # Disjunct union of dmg_mask and new_damage_mask is mask
    
    dmg_ds = WildfireDataset( # Dataset constructed only for reconstruction purpose
        dmg_x.float(),
        years.long(),
        months.long(),
        continuous_time.float(),
        x[:, :2].float(),
        dmg_mask.float(),
        new_damage_mask.float(),
    )
    reconstruct_valid_ds = Subset(dmg_ds, valid_idx)  # Reconstruction is made only on validation set, as training is "seen" by the model explicitly

    
    # THIRD. Create and return Dataloaders object
    
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=batch_size, drop_last=False)
    reconstruct_valid_dl = DataLoader(reconstruct_valid_ds, shuffle=False, batch_size=batch_size, drop_last=False)
    all_dl = DataLoader(all_ds, shuffle=False, batch_size=batch_size, drop_last=False)
        
    return (
        DataLoaders(train_dl, valid_dl, reconstruct_valid_dl, all_dl, device=torch.device(device)),
        mean,
        std,
        master_mask,
        valid_years, # also return valid_years
    )
