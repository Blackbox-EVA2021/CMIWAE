rm(list = ls())
getwd()
dir()

# read data:
load("data_train.RData")

### Some small inspection of the data
str(data_train_DF)
head(data_train_DF)

### Master mask
latitudes = sort(unique(data_train_DF$lat), decreasing = TRUE)
longitudes = sort(unique(data_train_DF$lon))

master_mask = matrix(rep(FALSE, length(latitudes) * length(longitudes)),
                     nrow = length(latitudes))

### This is a matrix with 2 columns that has all the pairs
### of lat-lon that are part of the USA
coordinates = unique(cbind(data_train_DF$lat, data_train_DF$lon))

for(i in 1:nrow(coordinates)){
  row_ind = which(coordinates[i, 1] == latitudes)
  col_ind = which(coordinates[i, 2] == longitudes)
  master_mask[row_ind, col_ind] = TRUE
}

### vectors of years and months
years_vec = rep(1993:2015, rep(7, 23))
months_vec = rep(3:9, 23)


### NA mask
NA_mask_CNT = array(dim = c(length(years_vec), dim(master_mask)))
NA_mask_BA = array(dim = c(length(years_vec), dim(master_mask)))

for(i in 1:length(years_vec)){
  NA_mask_CNT[i, , ] = master_mask
  NA_mask_BA[i, , ] = master_mask
}

for(r in 1:nrow(data_train_DF)){
  year_ind = data_train_DF$year[r]
  month_ind = data_train_DF$month[r]
  lat_ind = data_train_DF$lat[r]
  lon_ind = data_train_DF$lon[r]
  time_ind = which((years_vec == year_ind) & (months_vec == month_ind))
  row_ind = which(latitudes == lat_ind)
  col_ind = which(longitudes == lon_ind)
  NA_mask_CNT[time_ind, row_ind, col_ind] = !is.na(data_train_DF$CNT[r])
  NA_mask_BA[time_ind, row_ind, col_ind] = !is.na(data_train_DF$BA[r])
}


### make tensor with all the data
data_tensor = array(dim = c(length(years_vec), ncol(data_train_DF),
                            dim(master_mask)))

data_train_DF$year = as.numeric(data_train_DF$year)
data_train_DF$month = as.numeric(data_train_DF$month)


row_to_tensor = matrix(rep(0, 3 * nrow(data_train_DF)), ncol = 3)
tensor_to_row = array(dim = c(length(years_vec), dim(master_mask)))

for(r in 1:nrow(data_train_DF)){
  if((r %% 10000) == 0)
    print(r)
  year_ind = data_train_DF$year[r]
  month_ind = data_train_DF$month[r]
  lat_ind = data_train_DF$lat[r]
  lon_ind = data_train_DF$lon[r]
  time_ind = which((years_vec == year_ind) & (months_vec == month_ind))
  row_ind = which(latitudes == lat_ind)
  col_ind = which(longitudes == lon_ind)
  for(c in 1:ncol(data_train_DF)){
    data_tensor[time_ind, c, row_ind, col_ind] = data_train_DF[r, c]
  }
  row_to_tensor[r, ] = c(time_ind, row_ind, col_ind)
  tensor_to_row[time_ind, row_ind, col_ind] = r
}

save(latitudes, longitudes, coordinates, master_mask,
     years_vec, months_vec,
     NA_mask_CNT, NA_mask_BA,
     data_tensor, row_to_tensor, tensor_to_row,
     file = "data_tensors.RData")