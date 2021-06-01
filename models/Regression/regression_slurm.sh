#!/bin/bash
for i in 0 1 2 3 4 
do
#${#array[@]}获取数组长度用于循环

    output="re_fi${i}"
    echo "#!/bin/bash" >slurm/"${output}.slurm"
    echo "#SBATCH --job-name=${output}" >>slurm/"${output}.slurm"
    echo "#SBATCH -o out/${output}.out" >>slurm/"${output}.slurm"
    echo "#SBATCH --partition=gpu_7d1g" >>slurm/"${output}.slurm"
    echo "#SBATCH --nodes=1" >>slurm/"${output}.slurm"
    echo "#SBATCH --ntasks-per-node=6" >>slurm/"${output}.slurm"
    echo "#SBATCH --mem=92000M" >>slurm/"${output}.slurm"
    echo "#SBATCH --time=96:00:00" >>slurm/"${output}.slurm"
    echo "#SBATCH --qos=normal" >>slurm/"${output}.slurm"
    echo "#SBATCH --gres=gpu:1" >>slurm/"${output}.slurm"
    
    echo "echo job start time is \`date\`" >>slurm/"${output}.slurm"
    echo "echo \`hostname\`" >>slurm/"${output}.slurm"

    echo "singularity exec --nv /home/baikliang2/singularityBox/simpletransformer.sig python3 regressionSlurmFold.py \\" >>slurm/"${output}.slurm"
    echo "--output_dir=\"out_models/finalversion0517$i\" \\" >>slurm/"${output}.slurm"
    echo "--bert_model=\"chinese_roberta_wwm_ext_pytorch\" \\" >>slurm/"${output}.slurm"
    echo "--eval_file='results/final_advice0517${i}.csv' \\" >>slurm/"${output}.slurm"
    echo "--finetune_data='data/final_regression_0517/fold$i' \\" >>slurm/"${output}.slurm"
    echo "--advice_data='data/final_regression_0517' \\" >>slurm/"${output}.slurm"
    echo "--train_epochs=15 " >>slurm/"${output}.slurm"
    echo "echo job end time is \`date\` " >>slurm/"${output}.slurm"
    
    sbatch slurm/"${output}.slurm"
        



done
