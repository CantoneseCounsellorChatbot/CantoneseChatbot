python train.py \
    --data_dir data/BTS_re_data_6k\
    --bert_model pretrained_model/bert-base-chinese\
    --GPU_index "0,1"\
    --train_batch_size 64\
    --num_train_epochs 20\
    --print_every 50\
    --output_dir output_re_6k

