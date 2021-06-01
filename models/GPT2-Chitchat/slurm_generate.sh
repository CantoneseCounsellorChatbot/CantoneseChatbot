#!/bin/bash


for i in 0 1 2 3 4
do
#${#array[@]}获取数组长度用于循环
    output="GPTmix$i"
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

    echo "singularity exec --nv /home/baikliang2/singularityBox/ubuntu18_py1.6_cuda10.2.sif python3 train.py  \\" >>"slurm/${output}.slurm"
     # training epochs
    echo "--epochs 30 \\" >>"slurm/${output}.slurm"
    echo "--batch_size 32   \\" >>"slurm/${output}.slurm"
    echo "--pretrained_model dialogue_model_40epoch  \\" >>"slurm/${output}.slurm"
    echo "--dialogue_model_output_path  out_models/modelGPT_mix_6ka$i  \\" >>"slurm/${output}.slurm"
    # the path of the trainingset
    echo "--train_raw_path data/GPT_combination/fold$i/train.txt  \\" >>"slurm/${output}.slurm"
    echo "--train_tokenized_path data/combination_tokenized$i.txt  \\" >>"slurm/${output}.slurm"
    echo "--device 0  \\" >>"slurm/${output}.slurm"
    echo "--num_workers 6  \\" >>"slurm/${output}.slurm"
    echo "--raw  " >>"slurm/${output}.slurm"

    echo "echo job end time is \`date\` " >>"slurm/${output}.slurm"
    sbatch "slurm/${output}.slurm"

        
        



done
