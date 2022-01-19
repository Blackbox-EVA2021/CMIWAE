#!/bin/bash

# Parameter $1 is number of run.
# Parameter $2 is starting prediction_no
# Parameter $3 is ending prediction_no
# Parameter $4 is cuda_device
# Parameter $5 is N
# Parameter $6 is K_sampling

echo Starting computation...
for ((p=$2;p<=$3;p++))
do
echo Run: $1, prediction: $p ...
python make_full_ensemble_prediction.py $1 $p $4 $5 $6
done
