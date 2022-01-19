#! /usr/bin/env python


import rpy2.robjects as robjects
from rpy2.robjects import FloatVector


def compute_scores(pred_fname):
    robjects.r['source']('compute_scores.R') # Loading the function we have defined in R.
    compute_scores_function_r = robjects.globalenv['compute_scores'] # Reading and processing data
    scores = compute_scores_function_r(pred_fname) #Invoking the R function and getting the result
    return list(scores)

def compute_multiple_scores(pred_fnames):
    robjects.r['source']('compute_scores.R') # Loading the function we have defined in R.
    compute_multiple_scores_function_r = robjects.globalenv['compute_multiple_scores'] # Reading and processing data
    pred_fnames_vect = robjects.StrVector(pred_fnames)
    scores = compute_multiple_scores_function_r(pred_fnames_vect) #Invoking the R function and getting the result
    return list(scores)