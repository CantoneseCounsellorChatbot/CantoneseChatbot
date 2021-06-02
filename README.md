# How to run
1. Clone this repository
2. Obtain training data and singularity images from Google Drive
3. Open terminal and navigate to the desired file (for instance ```Regression```)
	```
	cd <pathname where you store the scripts>/models/Regression
	```
4. Adjust the parameters in either the .slurm/.sh files
   - For single-run training, use ```sample.sh``` under the ```slurm``` folder
   - For multiple runs (for instance when carrying out cross validation), use ```regression_slurm.sh```
	```
	#!/bin/bash
	singularity exec --nv <pathname where you store the singularity images> python3 regressionSlurmFold.py \
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

5. Begin training by inputting `bash slurm/sample.sh` or `bash regression_slurm.sh` into the terminal

6. The training progress could be checked from the output files in the `out/` folder, which is generated upon successful execution (for instance, use `cat out/regression0.out` to check the training progress of the regression model)
   
