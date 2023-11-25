#!/bin/bash

rm model/PPMI/GlobalModel.txt
python PPMI_prediction_NN.py blabla bla >> results.txt

for i in $(seq 1 5);
do
    python PPMI_prediction_NN.py
    
    echo q | python Split_Aggregate.py
    cd build
    ./fedavg_aggregation -r 0 -n 100 -d "PPMI"&
    ./fedavg_aggregation -r 1 -n 100 -d "PPMI"
    cd ..
    echo c | python Split_Aggregate.py

    python PPMI_prediction_NN.py blabla >> results.txt

    echo $i >> results.txt

done
