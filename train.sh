for lr in 0.001 0.0001 0.00001
do
    for batch in 8 16 32 64
    do
        for len in 64 128
        do
            nohup python3 train.py --lr ${lr} --batch_size ${batch} --max_length ${len} &> logs/logs-${batch}-${len}.txt
        done
    done
done 
