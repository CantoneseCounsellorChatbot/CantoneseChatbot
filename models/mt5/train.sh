#!/bin/bash


for i in 0 1 2 3 4
do
#${#array[@]}获取数组长度用于循环
    output="mtcq$i"
    echo "#!/bin/bash" >"slurm/${output}.slurm"
    echo "#SBATCH --job-name=${output}" >>"slurm/${output}.slurm"
    echo "#SBATCH -o out/${output}.out" >>"slurm/${output}.slurm"
    echo "#SBATCH --partition=gpu_7d1g" >>"slurm/${output}.slurm"
    echo "#SBATCH --nodes=1" >>"slurm/${output}.slurm"
    echo "#SBATCH --ntasks-per-node=6" >>"slurm/${output}.slurm"
    echo "#SBATCH --mem=92000M" >>"slurm/${output}.slurm"
    echo "#SBATCH --time=96:00:00" >>"slurm/${output}.slurm"
    echo "#SBATCH --qos=normal" >>"slurm/${output}.slurm"
    echo "#SBATCH --gres=gpu:1" >>"slurm/${output}.slurm"
    
    echo "echo job start time is \`date\`" >>"slurm/${output}.slurm"
    echo "echo \`hostname\`" >>"slurm/${output}.slurm"

    echo "PYTHONIOENCODING=utf-8 singularity exec --nv /home/baikliang2/singularityBox/mt5_2.sig python3 mt5_finetune.py    \\" >>"slurm/${output}.slurm"
    echo "--train_epochs 10 \\" >>"slurm/${output}.slurm"
    echo "--train_batch_size 8   \\" >>"slurm/${output}.slurm"
    echo "--valid_batch_size 8  \\" >>"slurm/${output}.slurm"
    echo "--train_data 'data/mT5_multi_sorted/train$i.csv'  \\" >>"slurm/${output}.slurm"
    echo "--test_data 'data/mT5_multi_sorted/test$i.csv'  \\" >>"slurm/${output}.slurm"
    echo "--save_model 'save_model/closed$i/'  \\" >>"slurm/${output}.slurm"
    echo "--save_file 'results/closedQtest$i.csv'  " >>"slurm/${output}.slurm"

    echo "echo job end time is \`date\` " >>"slurm/${output}.slurm"
    sbatch "slurm/${output}.slurm"

        
        



done
