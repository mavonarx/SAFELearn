#!/bin/bash

rm results.txt
rm model/PPMI/GlobalModel.txt
python PPMI_prediction_NN.py 2 999  >> results.txt

for i in $(seq 1 5);
do

    echo $i >> results.txt
    python PPMI_prediction_NN.py 0 $i
    
    echo q | python Split_Aggregate.py
    cd build
    ./fedavg_aggregation -r 0 -n 100 -d "PPMI"&
    ./fedavg_aggregation -r 1 -n 100 -d "PPMI"
    cd ..
    echo c | python Split_Aggregate.py

    python PPMI_prediction_NN.py 1 $i >> results.txt

done
