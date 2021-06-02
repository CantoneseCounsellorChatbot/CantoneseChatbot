#!/bin/bash


for i in 0 1 2 3 4 
do
#${#array[@]}获取数组长度用于循环
    output="GPTmix$i"
    echo "#!/bin/bash" >"${output}.slurm"
    echo "#SBATCH --job-name=${output}" >>"${output}.slurm"
    echo "#SBATCH -o out${output}.out" >>"${output}.slurm"
    echo "#SBATCH --partition=gpu_7d1g" >>"${output}.slurm"
    echo "#SBATCH --nodes=1" >>"${output}.slurm"
    echo "#SBATCH --ntasks-per-node=6" >>"${output}.slurm"
    echo "#SBATCH --mem=92000M" >>"${output}.slurm"
    echo "#SBATCH --time=96:00:00" >>"${output}.slurm"
    echo "#SBATCH --qos=normal" >>"${output}.slurm"
    echo "#SBATCH --gres=gpu:1" >>"${output}.slurm"
    
    echo "echo job start time is \`date\`" >>"${output}.slurm"
    echo "echo \`hostname\`" >>"${output}.slurm"

    echo "singularity exec --nv /home/baikliang2/singularityBox/ubuntu18_py1.6_cuda10.2.sif python3 interact_copy_0221.py  \\" >>"${output}.slurm"
    # the path of fintune model
    echo "--dialogue_model_path out_models/modelGPT_mix_6ka$i/model_epoch30 \\" >>"${output}.slurm"
    echo "--max_history_len 1   \\" >>"${output}.slurm"
    # Save path of results
    echo "--result_path outputs/GPT6kamix${i}.txt  \\" >>"${output}.slurm"
    # path of test set
    echo "--test_path data/GPT_combination/fold$i/test.txt  \\" >>"${output}.slurm"
    # maximun length of the generation reply
    echo "--max_len 50  " >>"${output}.slurm"

    echo "echo job end time is \`date\` " >>"${output}.slurm"
    sbatch "${output}.slurm"
        
        
        



done
