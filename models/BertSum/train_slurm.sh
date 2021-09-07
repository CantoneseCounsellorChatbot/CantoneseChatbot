#!/bin/bash


for i in 0 1 2 3 4 
do
    output="B_r+p$i"
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

    echo "singularity exec --nv /home/baikliang2/singularityBox/BertAbsSum.sig python3 train.py  \\" >>"slurm/${output}.slurm"
    # the path of training set and test set
    echo "--data_dir data/BTS_re+pr/fold$i \\" >>"slurm/${output}.slurm"
    # the path of pretrained model
    echo "--bert_model pretrained_model/bert-base-chinese   \\" >>"slurm/${output}.slurm"
    echo "--GPU_index 0,1  \\" >>"slurm/${output}.slurm"
    echo "--train_batch_size 64 \\" >>"slurm/${output}.slurm"
    echo "--num_train_epochs 15  \\" >>"slurm/${output}.slurm"
    echo "--print_every 500  \\" >>"slurm/${output}.slurm"
    # The save path of the model
    echo "--output_dir outputmodels/output_BTS_re+pr$i " >>"slurm/${output}.slurm"

    echo "echo job end time is \`date\` " >>"slurm/${output}.slurm"
    sbatch "slurm/${output}.slurm"
        
        
        



done