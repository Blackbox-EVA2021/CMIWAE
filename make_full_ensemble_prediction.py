#! /usr/bin/env python


from make_prediction import (
    save_observation_data,
    make_dataloader,
    create_model_name,
    load_CNT_and_BA_unknown_idx,
    make_predictions_from_data_improved,
    load_params_from_command_line,
    create_model_for_run,
)
from statistics_helper import get_all_models


if __name__ == "__main__":
    # Code to produce prediction
    run, fpath, prediction_no, device, N, K_sampling = load_params_from_command_line()

    print(f"Computing prediction {prediction_no} for run {run} on {device} ...")
    
    # Save missing unknown data for later scoring in observation.RData #
    CNT_unknown_idx_y, BA_unknown_idx_y = load_CNT_and_BA_unknown_idx(device=device)
    save_observation_data(CNT_unknown_idx_y, BA_unknown_idx_y)
    ####################################################################
    
    model, model_params = create_model_for_run(run, device)
    
    model_fnames, _ = get_all_models(fpath=fpath, get_model_name_fcn=create_model_name, **model_params)
    del model_params
    print(f"Using full {len(model_fnames)} models for prediction {prediction_no} ...")
    print(model_fnames[:3])
    
    test_dl = make_dataloader(device=device)
    assert len(test_dl) == 161  # HARDCODED total number of months
    
    CNT_unknown_idx, BA_unknown_idx = load_CNT_and_BA_unknown_idx(device=device, load_data_per_year=False)
          
    test_years = list(range(1,23,2))  # competition prediction years
    
    models_per_year = {}
    for y in test_years:
        models_per_year[y] = model_fnames  # Use all models for every test year
    
    make_predictions_from_data_improved(model, models_per_year, test_years, test_dl, CNT_unknown_idx, BA_unknown_idx,
                                        N=N, default_K=K_sampling,
                                        device=device,
                                        rdata_filename=f"predictions/run{run}_prediction{prediction_no}_N{N}_K{K_sampling}_full.RData")