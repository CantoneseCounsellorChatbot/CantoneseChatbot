#!/bin/bash


for i in 0 1 2 3 4
do
#${#array[@]}获取数组长度用于循环
    output="cq_mmi$i"
    echo "#!/bin/bash" >"slurm/${output}.slurm"
    echo "#SBATCH --job-name=${output}" >>"slurm/${output}.slurm"
    echo "#SBATCH -o out/${output}.out" >>"slurm/${output}.slurm"
    echo "#SBATCH --partition=gpu_7d1g" >>"slurm/${output}.slurm"
    echo "#SBATCH --nodes=1" >>"slurm/${output}.slurm"
    echo "#SBATCH --ntasks-per-node=6" >>"slurm/${output}.slurm"
    echo "#SBATCH --mem=92000M" >>"slurm/${output}.slurm"
    echo "#SBATCH --time=30:00:00" >>"slurm/${output}.slurm"
    echo "#SBATCH --qos=normal" >>"slurm/${output}.slurm"
    echo "#SBATCH --gres=gpu:1" >>"slurm/${output}.slurm"
    
    echo "echo job start time is \`date\`" >>"slurm/${output}.slurm"
    echo "echo \`hostname\`" >>"slurm/${output}.slurm"

    echo "singularity exec --nv /home/baikliang2/singularityBox/ubuntu18_py1.6_cuda10.2.sif python3 train.py  \\" >>"slurm/${output}.slurm"
    echo "--epochs 40 \\" >>"slurm/${output}.slurm"
    echo "--batch_size 32   \\" >>"slurm/${output}.slurm"
    echo "--pretrained_model out_models/modelGPT_multi-closedQ$i  \\" >>"slurm/${output}.slurm"
    echo "--mmi_model_output_path  out_models/cq6kmultiple_mmi$i  \\" >>"slurm/${output}.slurm"
    echo "--train_raw_path data/multiCQ_DialoGPT5fold_6k/fold$i/train.txt  \\" >>"slurm/${output}.slurm"
    echo "--train_mmi_tokenized_path data/mmi_tokenized$i.txt  \\" >>"slurm/${output}.slurm"
    echo "--device 0  \\" >>"slurm/${output}.slurm"
    echo "--num_workers 6  \\" >>"slurm/${output}.slurm"
    echo "--train_mmi  \\" >>"slurm/${output}.slurm"
    echo "--raw  " >>"slurm/${output}.slurm"

    echo "echo job end time is \`date\` " >>"slurm/${output}.slurm"
    sbatch "slurm/${output}.slurm"
        
        
        



done
