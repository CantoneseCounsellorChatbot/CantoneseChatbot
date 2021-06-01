import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy.special import softmax

from simpletransformers.classification import ClassificationModel

import sklearn
import logging
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--output_dir",
                    default='outputs',
                    type=str,
                    help="the output path for models")



parser.add_argument("--bert_model",
                    default=None,
                    type=str,
                    help="The pretrained model path")
parser.add_argument("--eval_file",
                    default=None,
                    type=str,
                    help="the output evaluation file")
parser.add_argument("--train_epochs",
                    default=30,
                    type=int,
                    help="training epoch")
parser.add_argument("--finetune_data",
                    default=None,
                    type=str,
                    help="the path of training data and test data")
parser.add_argument("--advice_data",
                    default=None,
                    type=str,
                    help="the path of candidate advice")




targs = parser.parse_args()




logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
EVAL_NUM=2000
np.random.seed(5)
file_name = targs.finetune_data


# for lbk in range(5):
advice_list=pd.read_csv(targs.advice_data+"/adviceall.csv").advice.to_list()
test_set = pd.read_csv(file_name+"/test.csv")
train_set = pd.read_csv(file_name+"/train.csv")
train1=train_set[train_set.labels==1]

tmp_train = train_set.copy()
tmp_test  = test_set.copy()


tmp_set = train_set
eval_set=tmp_set.iloc[:EVAL_NUM,:]
train_set = tmp_set.iloc[EVAL_NUM:,:]
train_df = train_set

eval_df = eval_set

train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': targs.train_epochs,
    'save_steps':-1,
    'save_model_every_epoch':False,
    'evaluate_during_training':True,
    'evaluate_during_training_steps':240000,
    "dataloader_num_workers":6,
    "use_multiprocessing":False,
    "output_dir":targs.output_dir,
    "best_model_dir":targs.output_dir+"/bestmodel",
    "train_batch_size":32,
    "regression":True
}
fold_bert_model = targs.bert_model
# fold_bert_model = targs.bert_model

# Create a ClassificationModel
model = ClassificationModel('bert', fold_bert_model, num_labels=1, use_cuda=True, cuda_device=0, args=train_args)
# print(train_df.head())

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.mean_squared_error)
model = ClassificationModel("bert", targs.output_dir+"/bestmodel" )

train_set = tmp_train
test_set = tmp_test
test_list=[]
test_list_100=[]
# advice_list= np.array(list(set(train1["text_b"].to_list())))
# symptom_list=np.array(alldata["symptoms"].to_list())

test_sentence=np.array(test_set["text_a"].drop_duplicates().to_list())
for index,row in test_set.iterrows():
    
    np.random.shuffle(advice_list)
    advice_list_100=np.append(advice_list,row["text_b"])
    advice_list_100 = advice_list_100[-100:]
    np.random.shuffle(advice_list_100)
    np.random.shuffle(advice_list)
    tmp=[]
    for advice in advice_list:
        tmp.append([row["text_a"],advice])
    test_list.append(tmp)
    tmp=[]
    for advice in advice_list_100:
        tmp.append([row["text_a"],advice])
    test_list_100.append(tmp)
topk=10
best_match = []
best_match_label=[]
best_score=[]
for i in tqdm(range(len(test_list))):
    predictions, raw_outputs = model.predict(test_list[i])
    topindex=predictions.argsort()[-topk:][::-1]
    best_match.append(np.array(test_list[i])[:,1][topindex])
    best_score.append(max(predictions))
for i in range(topk):
    test_set["match"+str(i)]=[x[i] for x in best_match]
test_set["score"]=best_score
# test_set.to_csv("1230NSP_bert_fold"+str(lbk)+".csv",encoding="utf_8_sig")

topk=10
best_match = []
best_match_label=[]
for i in tqdm(range(len(test_list))):
    predictions, raw_outputs = model.predict(test_list_100[i])
    topindex=predictions.argsort()[-topk:][::-1]
    best_match.append(np.array(test_list_100[i])[:,1][topindex])
for i in range(topk):
    test_set["100match"+str(i)]=[x[i] for x in best_match]
test_set.to_csv(targs.eval_file,encoding="utf_8_sig")