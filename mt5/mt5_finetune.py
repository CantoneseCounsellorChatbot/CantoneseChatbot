import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from Rouge_BLEU_corpus import Rouge_BLEU_multiple
import os


# Importing the T5 modules from huggingface/transformers
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# WandB – Import the wandb library
import wandb
from torch import cuda
# device="cpu"
device = 'cuda' if cuda.is_available() else 'cpu'
def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', default=10, type=int, required=False, help='训练的轮次')
    parser.add_argument('--train_batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--valid_batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1e-4, type=float, required=False, help='学习率')
    parser.add_argument('--summary_len', default=50, type=int, required=False, help='warm up步数')
    parser.add_argument('--val_epochs', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--test_data', default="", type=str, required=False, help='多少步汇报一次loss')
    parser.add_argument('--train_data', default="", type=str, required=False, help='多少步汇报一次loss')
    parser.add_argument('--save_file', default="", type=str, required=False, help='多少步汇报一次loss')
    parser.add_argument('--save_model', default="", type=str, required=False, help='多少步汇报一次loss')

    # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
    # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
    return parser.parse_args()

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})
            # pass

        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def main():
    # WandB – Initialize a new run
    args=setup_train_args()
    wandb.login(key="e5922292a799ae407ea4f5a113a8874d8e5a24e2")
    wandb.init(project="closedQ")
    

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training  
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = args.train_batch_size    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = args.valid_batch_size    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = args.train_epochs        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = args.val_epochs
    config.LEARNING_RATE = args.lr   # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 512
    config.SUMMARY_LEN = args.summary_len 
    # config = {}          # Initialize config
    # config["TRAIN_BATCH_SIZE"] = args.train_batch_size    # input batch size for training (default: 64)
    # config["VALID_BATCH_SIZE"] = args.valid_batch_size    # input batch size for testing (default: 1000)
    # config["TRAIN_EPOCHS"] = args.train_epochs        # number of epochs to train (default: 10)
    # config["VAL_EPOCHS"] = args.val_epochs
    # config["LEARNING_RATE"] = args.lr   # learning rate (default: 0.01)
    # config["SEED"] = 42               # random seed (default: 42)
    # config["MAX_LEN"] = 512
    # config["SUMMARY_LEN"] = args.summary_len 

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    df = pd.read_csv(args.train_data,encoding='utf-8')
    # df=df.iloc[:100,:]
    df = df[['text','ctext']]
    eval_set=df.iloc[:1000,]
    t_eval_post=eval_set.ctext.str.replace("summarize:   ","").to_list()
    df=df.iloc[1000:,:]
    df.reset_index(drop=True, inplace=True)
#     df.ctext = 'summarize: ' + df.ctext
    print(df.head())

    
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation. 
    # train_size = 0.8
    train_dataset=df
    val_dataset=pd.read_csv(args.test_data,encoding='utf-8')

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    t_eval_set=CustomDataset(eval_set, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }
    t_eval_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    t_eval_loader = DataLoader(t_eval_set, **t_eval_params)


    
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')
    rouge_list=[]
    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        predictions, actuals = validate(1, tokenizer, model, device, t_eval_loader)
        final_df = pd.DataFrame({"post":t_eval_post,'hyp':predictions,'ref':actuals})
        score = Rouge_BLEU_multiple(final_df)
        rouge_list.append(score)
        if score == max(rouge_list):
            if not os.path.exists(args.save_model+"epoch"+str(epoch)+"/"):
                os.makedirs(args.save_model+"epoch"+str(epoch)+"/")
            model.save_pretrained(args.save_model+"epoch"+str(epoch)+"/")
            # final_df.to_csv(args.save_file+str(epoch), index=False)


    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    best_epoch = np.argmax(rouge_list)

    model = MT5ForConditionalGeneration.from_pretrained(args.save_model+"epoch"+str(best_epoch)+"/")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(config.VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"post":val_dataset.ctext.str.replace("summarize:   ","").to_list(),'hyp':predictions,'ref':actuals})
        final_df.to_csv(args.save_file)
        print('Output Files generated for review')

if __name__ == '__main__':
    main()
