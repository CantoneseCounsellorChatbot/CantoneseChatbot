PYTHONIOENCODING=utf-8 singularity exec --nv /home/kent/newhome/singularityBox/mt5 python3 mt5_finetune.py  \
--train_epochs 10 \
--train_batch_size 2 \
--valid_batch_size 2 