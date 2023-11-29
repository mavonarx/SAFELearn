#!/bin/bash

rm results.txt
rm model/PPMI/GlobalModel.txt
echo "NO safelearn" >> results.txt
python PPMI_prediction_NN.py 2 999  >> results.txt

for i in $(seq 1 5);
do

    echo $i >> results.txt
    python PPMI_prediction_NN.py 0 $i
    python Python_FL_aggregation.py

    python PPMI_prediction_NN.py 1 $i >> results.txt
    

done
