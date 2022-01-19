#!/bin/bash

# Parameter $1 is number of run.
# Parameter $2 is beg model idx
# Parameter $3 is end model idx

echo Starting computation...
for ((idx=$2;idx<=$3;idx++))
do
echo Run: $1, idx: $idx ...
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m fastai.launch train_run_$1.py $idx $1 > outputs/output-run$1-idx$idx.txt
done
