#!/bin/bash


for i in 0 1 2 3 4
do
#${#array[@]}获取数组长度用于循环
    output="pre_com$i"
    echo "#!/bin/bash" >"slurm/${output}.slurm"
    echo "#SBATCH --job-name=${output}" >>"slurm/${output}.slurm"
    echo "#SBATCH -o out/${output}.out" >>"slurm/${output}.slurm"
    echo "#SBATCH --partition=gpu_1d2g" >>"slurm/${output}.slurm"
    echo "#SBATCH --nodes=1" >>"slurm/${output}.slurm"
    echo "#SBATCH --ntasks-per-node=6" >>"slurm/${output}.slurm"
    echo "#SBATCH --mem=92000M" >>"slurm/${output}.slurm"
    echo "#SBATCH --time=24:00:00" >>"slurm/${output}.slurm"
    echo "#SBATCH --qos=normal" >>"slurm/${output}.slurm"
    echo "#SBATCH --gres=gpu:2" >>"slurm/${output}.slurm"
    
    echo "echo job start time is \`date\`" >>"slurm/${output}.slurm"
    echo "echo \`hostname\`" >>"slurm/${output}.slurm"


    echo "singularity exec --nv /home/baikliang2/singularityBox/BertAbsSum.sig python3  predict.py  \\" >>"slurm/${output}.slurm"
    echo "--model_path outputmodels/output_BTS_Combination$i/model/BertAbsSum_14.bin \\" >>"slurm/${output}.slurm"
    echo "--config_path outputmodels/output_BTS_Combination$i/model/config.json   \\" >>"slurm/${output}.slurm"
    echo "--eval_path data/BTS_Combination/fold$i/test.csv \\" >>"slurm/${output}.slurm"
    echo "--bert_model pretrained_model/bert-base-chinese  \\" >>"slurm/${output}.slurm"
    # Save path of results
    echo "--result_path results/BTS_combination_allmultiresult$i " >>"slurm/${output}.slurm"


    echo "echo job end time is \`date\` " >>"slurm/${output}.slurm"
    sbatch "slurm/${output}.slurm"
        
        
        



done
