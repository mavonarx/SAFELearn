python3  -u main.py --dataset=sent140 --optimizer=qffedavg  \
            --learning_rate=0.03 \
            --learning_rate_lambda=0.01 \
            --num_rounds=1 \
            --eval_every=1 \
            --clients_per_round=10 \
            --batch_size=32 \
            --q=1 \
            --model='svm' \
            --sampling=1  \
            --num_epochs=10 \
            --data_partition_seed=1 \
            --log_interval=10 \
            --static_step_size=0 \
            --track_individual_accuracy=0 #\
            #--output="./log_$1/$2_samp$5_run$3_q$4"