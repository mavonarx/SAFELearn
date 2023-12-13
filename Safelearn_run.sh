#!/bin/bash

rm results.txt
rm model/PPMI/GlobalModel.txt


# Change params here
##########################################################
# mode 2 = Q-fed-avg, mode 1 = weighted-avg, mode 0 = normal avg
MODE=2
ROUNDS=20
##########################################################
echo "SAFELearn" >> results.txt
echo MODE = $MODE >> results.txt
echo ROUNDS = ${ROUNDS} >> results.txt

if [ "$MODE" -eq 2 ]; then
    SPLIT_STR="q"
else
    SPLIT_STR="s"
fi 

for i in $(seq 1 ${ROUNDS});
do
    echo $i >> results.txt
    cd fair_flearn
    ./run.sh 
    cd ..
    echo $SPLIT_STR | python Split_Aggregate_TF.py
    cd build
    ./fedavg_aggregation -q $MODE -r 0 -n 100 -d "PPMI" 1>/dev/null &
    ./fedavg_aggregation -q $MODE -r 1 -n 100 -d "PPMI"
    cd ..
    echo c | python Split_Aggregate_TF.py

    cat fair_flearn/vehicle_stuff_to_cat/test.csv >> results.txt
    cat fair_flearn/vehicle_stuff_to_cat/train.csv >> results.txt
    cat fair_flearn/vehicle_stuff_to_cat/val.csv >> results.txt
done
