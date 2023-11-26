#!/bin/bash

rm results.txt
rm model/PPMI/GlobalModel.txt
python PPMI_prediction_NN.py 2 999  >> results.txt

# Change params here
##########################################################
# mode 2 = Q-fed-avg, mode 1 = weighted-avg, mode 0 = normal avg
MODE=2
##########################################################


if ["$MODE" -eq 2 ]; then
    SPLIT_STR="q"
else
    SPLIT_STR="s"
fi 

for i in $(seq 1 5);
do

    echo $i >> results.txt
    python PPMI_prediction_NN.py 0 $i
    
    echo ($SPLIT_STR) | python Split_Aggregate.py
    cd build
    ./fedavg_aggregation -q ($MODE) -r 0 -n 100 -d "PPMI"&
    ./fedavg_aggregation -q ($MODE) -r 1 -n 100 -d "PPMI"
    cd ..
    echo c | python Split_Aggregate.py

    python PPMI_prediction_NN.py 1 $i >> results.txt

done
