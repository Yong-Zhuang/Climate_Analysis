#!/bin/bash

echo "Beginning Experiments..."

counter = 0

for nru in {2..14..2}
do
    for norm in {0..2}
    do      
        echo "Running experiment: ${counter}"
        echo "  Number of Residual Units: ${nru}"
        echo "  Normalization: ${norm}"
        python experiment.py --data 10 --mode 1 --nru $nru --norm $norm
        ((counter++))
    done
done
