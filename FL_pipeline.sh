#!/bin/bash

# Pipeline WITHOUT safelearn

# change params here
###################################################
ROUNDS=5
###################################################

rm results.txt
rm model/PPMI/GlobalModel.txt
echo "NO safelearn" >> results.txt
echo ROUNDS = $ROUNDS >> results.txt
python PPMI_prediction_NN.py 2 999  >> results.txt

for i in $(seq 1 ${ROUNDS});
do

    echo $i >> results.txt
    python PPMI_prediction_NN.py 0 $i
    python Python_FL_aggregation.py

    python PPMI_prediction_NN.py 1 $i >> results.txt
    

done
