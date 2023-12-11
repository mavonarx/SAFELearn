python3  -u main.py --dataset=ppmi --optimizer=qffedavg_ppmi  \
            --learning_rate=0.0001 \
            --learning_rate_lambda=0.01 \
            --num_rounds=1 \
            --eval_every=1 \
            --clients_per_round=5 \
            --batch_size=64 \
            --q=0.01 \
            --model='PPMI_TF_NN' \
            --sampling=1  \
            --num_epochs=100 \
            --data_partition_seed=1 \
            --log_interval=10 \
            --static_step_size=0 \
            --track_individual_accuracy=0 #\
            #--output="./log_$1/$2_samp$5_run$3_q$4"