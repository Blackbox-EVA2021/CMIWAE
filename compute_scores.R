get_score_cnt = function(prediction_cnt, obs, u_cnt, weights_cnt){
    distr_obs = c()
    for(k in 1:length(u_cnt)){
        distr_obs = cbind(distr_obs, ifelse(u_cnt[k] < obs, 0, 1))
    }
    weights_mat = matrix(weights_cnt, ncol = length(weights_cnt), nrow = length(obs), byrow = TRUE)
    score_cnt = sum(weights_mat * (distr_obs - prediction_cnt)^2)
    score_cnt
}

get_score_ba = function(prediction_ba, obs, u_ba, weights_ba){
    distr_obs = c()
    for(k in 1:length(u_ba)){
        distr_obs = cbind(distr_obs, ifelse(u_ba[k] < obs, 0, 1))
    }
    weights_mat = matrix(weights_ba, ncol = length(weights_ba), nrow = length(obs), byrow = TRUE)
    score_ba = sum(weights_mat * (distr_obs - prediction_ba)^2)
    score_ba
}

compute_scores = function(pred_fname){
    load("observation.RData")
    load("data_train.RData")
    
    load(pred_fname)
    score_cnt = get_score_cnt(prediction_cnt, obs_cnt, u_cnt, weights_cnt)
    score_ba = get_score_ba(prediction_ba, obs_ba, u_ba, weights_ba)
    
    scores = c(score_cnt, score_ba)
    scores
}

compute_multiple_scores = function(pred_fname_vect){
    load("observation.RData")
    load("data_train.RData")
    
    scores_vect = c()
    for(i in 1:length(pred_fname_vect)){
        pred_fname <- pred_fname_vect[i]
        load(pred_fname)
        score_cnt = get_score_cnt(prediction_cnt, obs_cnt, u_cnt, weights_cnt)
        score_ba = get_score_ba(prediction_ba, obs_ba, u_ba, weights_ba)
        scores_vect[i] <- score_cnt + score_ba
    }
    
    scores_vect
}