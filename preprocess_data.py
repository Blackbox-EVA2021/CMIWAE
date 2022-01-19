import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from matplotlib import pyplot as plt
numpy2ri.activate()


r_load = robjects.r['load']

print(r_load("data_tensors.RData"))

data_tensor, master_mask, NA_mask_CNT, NA_mask_BA, row_to_tensor, tensor_to_row = [np.array(
    robjects.r[name]) for name in ["data_tensor", "master_mask", "NA_mask_CNT", "NA_mask_BA", "row_to_tensor", "tensor_to_row"]]
row_to_tensor = (row_to_tensor - 1).astype(int).T
tensor_to_row = (tensor_to_row - 1).astype(int)


# # preprocess and save data



mask = np.ones(data_tensor.shape, dtype=bool)
np.copyto(mask, master_mask.astype(bool))
mask[:, 0, ...] = NA_mask_CNT
mask[:, 1, ...] = NA_mask_BA

masked_data_tensor = np.ma.array(data_tensor, mask=~mask)


# `0. CNT, leave as is
#
# `1. BA, leave as is
#
# `2. normalize
#
# `3. normalize
#
# `4. territory percentage, mostly ==1, leave as is
#
# `5. year, convert to continous variable
#
# `6. month, leave discrete, use as embedding
#
# `7 - 24. proportion of land cover classes, leave as is
#
# `25. height mean, normalize
#
# `26. height deviation, normalize
#
# `27 - 36. mean meteo variables, normalize


# # preprocess and save data


# throw out years, months
masked_data_tensor = np.ma.concatenate(
    [masked_data_tensor[:, :5], masked_data_tensor[:, 7:]], axis=1)


# normalize data
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    mean = np.ma.mean(masked_data_tensor, axis=(0, 2, 3), keepdims=True)
    std = np.ma.std(masked_data_tensor, axis=(0, 2, 3), keepdims=True)

# it seems most channels are special cases! :)

# we do not modify CNT and BA!

# CNT
mean[:, 0] = 0
std[:, 0] = 1

# BA
mean[:, 1] = 0
std[:, 1] = 1

# we threw out year, month at indices 5, 6!!!

# area fraction
mean[:, 4] = 0
std[:, 4] = 1

# proportion of land cover class
mean[:, 5:23] = 0
std[:, 5:23] = 1

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    normalized_masked = (masked_data_tensor - mean) / std


normalized = normalized_masked.filled(0)


BA_mask = NA_mask_BA.astype(bool)
CNT_mask = NA_mask_CNT.astype(bool)

# if BA is 0, assume CNT is 0 and vice versa
CNT_known_zeros = CNT_mask & (normalized[:, 0] == 0)
BA_known_zeros = BA_mask & (normalized[:, 1] == 0)


# resize spatial dims to closest power of 2


padded = np.zeros((161, 35, 64, 128))
padded[:, :, 8:8+49, 6:6+117] = normalized

padded_master_mask = np.zeros((64, 128))
padded_master_mask[8:8+49, 6:6+117] = master_mask

padded_BA_mask = np.zeros((161, 64, 128))
padded_BA_mask[:, 8:8+49, 6:6+117] = NA_mask_BA

padded_CNT_mask = np.zeros((161, 64, 128))
padded_CNT_mask[:, 8:8+49, 6:6+117] = NA_mask_CNT

padded_row_to_tensor = row_to_tensor + np.array([[0, 8, 6]]).T


master_mask_b = master_mask.astype(bool)


CNT_unknown_1d_bool_idx = ~(padded_CNT_mask.astype(bool)[
                            tuple(padded_row_to_tensor)])
BA_unknown_1d_bool_idx = ~(padded_BA_mask.astype(bool)[
                           tuple(padded_row_to_tensor)])


padded_CNT_unknown_idx = padded_row_to_tensor[:, CNT_unknown_1d_bool_idx]
padded_BA_unknown_idx = padded_row_to_tensor[:, BA_unknown_1d_bool_idx]




all_idx = list(range(161))
train_idx, valid_idx = train_test_split(
    all_idx, test_size=0.15, random_state=42+1)

np.savez("padded_normalized.npz",
         data=padded,
         master_mask=padded_master_mask,
         BA_mask=padded_BA_mask,
         CNT_mask=padded_CNT_mask,
         years=robjects.r["years_vec"],
         months=robjects.r["months_vec"],
         mean=mean,
         std=std,
         row_to_tensor=padded_row_to_tensor,
         train_idx=train_idx,
         valid_idx=valid_idx,
         CNT_unknown_idx=padded_CNT_unknown_idx,
         BA_unknown_idx=padded_BA_unknown_idx,
         )
