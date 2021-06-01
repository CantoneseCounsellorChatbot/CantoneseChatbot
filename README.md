1. Get everything from Github
2. Open terminal and navigate to the desired file (i.e. Regression)
	```
	cd .…/models/Regression
	```
3. Adjust the parameters in either the .slurm/.sh files
	 - For single-run training, use ```sample.sh``` under the ```slurm``` folder
   - For multiple runs (for instance when carrying out cross validation), use ```regression_slurm.sh```
	```
	#!/bin/bash
	singularity exec --nv /home/baikliang2/singularityBox/simpletransformer.sig python3 regressionSlurmFold.py \
	# the output path for models
	--output_dir="out_models/finalversion05170" \
	# The pretrained model path
	--bert_model="chinese_roberta_wwm_ext_pytorch" \
	# the output evaluation file
	--eval_file='results/final_advice05170.csv' \
	# the path of training data and test data
	--finetune_data='data/final_regression_0517/fold0' \
	# the path of candidate advices
	--advice_data='data/final_regression_0517' \
	--train_epochs=15 
	```

4. Begin training by inputting the following into the terminal
	 - For single-run training, input ```bash slurm/sample.sh```
   
   
