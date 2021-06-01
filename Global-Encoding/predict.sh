#!/bin/bash


for i in 0 1 2 3 4
do
#${#array[@]}获取数组长度用于循环
    output="pGEmix$i"
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

    echo "singularity exec --nv /home/baikliang2/singularityBox/GlobalEncoding.sif python3 train.py  \\" >>"slurm/${output}.slurm"
    echo "-log log_result \\" >>"slurm/${output}.slurm"
    echo "-config lcsts.yaml   \\" >>"slurm/${output}.slurm"
    echo "-gpus 0  \\" >>"slurm/${output}.slurm"
    echo "-restore output/GE_combination$i/log_lbk/best_rouge_lbk_checkpoint.pt  \\" >>"slurm/${output}.slurm"
    echo "-mode eval  \\" >>"slurm/${output}.slurm"
    echo "-data data/GE_combination/fold${i}/  \\" >>"slurm/${output}.slurm"
    echo "-logF results/GE_combination_6ka$i/  \\" >>"slurm/${output}.slurm"
    echo "-resultF results/result_GE_combination_6ka$i  " >>"slurm/${output}.slurm"

    echo "echo job end time is \`date\` " >>"slurm/${output}.slurm"
    sbatch "slurm/${output}.slurm"
        
        
        



done