#!/bin/bash

for i in $(seq 1 2);
do
    echo $i
    python PPMI_prediction_NN.py
    
    echo q | python Split_Aggregate.py
    cd build
    ./fedavg_aggregation -r 0 -n 100 -d "PPMI"&
    ./fedavg_aggregation -r 1 -n 100 -d "PPMI"
    cd ..
    echo c | python Split_Aggregate.py

    python PPMI_prediction_NN.py blabla
    echo $i
done