import sys
# from google.colab import drive
# drive.mount('/content/drive')
# sys.path.append('/content/drive/My Drive/chatbot/BertSum')
from datetime import time
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm
from simpletransformers.classification import ClassificationModel
from transformers import BertTokenizer, BertForSequenceClassification
# import sqlite3
import re
import matplotlib.pyplot as plt

def regressionReply(post,model,candidate):
  model = ClassificationModel("bert", model )
  data=pd.read_csv(candidate)
  advice_list = data.advice.drop_duplicates().to_list()      
  np.random.shuffle(advice_list)
  tmp=[]
  for advice in advice_list:
      tmp.append([post,advice])


  predictions, raw_outputs = model.predict(tmp)
  topindex=np.argmax(predictions)
  
  
  return tmp[topindex][1],max(predictions)

'''
def regressionReply(post,model,candidate):
  # print("regression")
  tokenizer = BertTokenizer.from_pretrained(model)
  model = BertForSequenceClassification.from_pretrained(model)
  model.to('cuda')
  data=pd.read_csv(candidate)
  advice_list = data.advice.drop_duplicates().to_list()
  text=post

  input_ids = []
  attention_masks = []
  for a in advice_list:
      encoded_dict = tokenizer.encode_plus(
                      text+"[SEP]"+a,                      # Sentence to encode.
                      add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                      max_length = 50,           # Pad & truncate all sentences.
                      pad_to_max_length = True,
                      return_attention_mask = True,   # Construct attn. masks.
                      return_tensors = 'pt',     # Return pytorch tensors.
                  )
      input_ids.append(encoded_dict['input_ids'])
      attention_masks.append(encoded_dict['attention_mask'])
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  batch_size = 32
  # print("chatbot:{}".format(advice))
  prediction_data = TensorDataset(input_ids, attention_masks)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
  
  predictions  = []
  # print("regression2")
  # Predict 
  for batch in tqdm(prediction_dataloader):
      batch = tuple(t.to("cuda") for t in batch)
      b_input_ids, b_input_mask = batch
      with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      logits = outputs[0]
      logits = logits.detach().cpu().numpy().reshape(-1).tolist()
  #     label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
      predictions+=logits
  #   true_labels.append(label_ids)

  return advice_list[np.argmax(predictions)], np.max(predictions)

'''


def general(aa,max_tail_length):
    def getQAlist():
        qaList = []
        exact_list=[]
        conn = pd.read_csv("/content/drive/My Drive/chatbot/keyword_list.csv")
        exact_match = conn[conn.KeywordMatch=="no"]
        conn=conn[conn.KeywordMatch=="yes"]
        for index,row in exact_match.iterrows():
            tmp = {"Q":""+row["Q"],"A":""+row["A"]}
            exact_list.append(tmp)

        
        
        for index,row in conn.iterrows():
          if row["Q"]!="*":
            tmp = {"Q":""+row["Q"],"A":""+row["A"]}
            qaList.append(tmp)
          else:
            genreal_reply=row["A"].split("|")
        return exact_list,qaList,genreal_reply

    def answer(say,seg):
        if (say[0]=="你") and (say.find("唔")>0):
            msg = handleSpecial(say,seg)
            if msg !="":
                return msg
        return "@eliza@ " + getAnswer(say)
    def getAnswer(say):
        exactmatch,tmpList,general_reply=getQAlist()
        results = analyzeSay(say, tmpList, general_reply,exactmatch)
        
        msg=results[1]
        if msg !="":
            return msg
        else:
            return "然後呢？@发生错误@"

    def analyzeSay(say, tmpList, general_reply,exactmatch):
        exact_df = pd.DataFrame(exactmatch)
        if say in exact_df.Q.to_list():
          return exact_df[exact_df.Q==say].A.item()
        patterns = []
        for i in range(len(tmpList)):
            qa = tmpList[i]
            qList = qa["Q"].split(" | ")
            aList = qa["A"].split("|")            
            elizakeyword = []
            for j in range(len(qList)):
                qi = qList[j]
 


                if say.find(qi) >-1:


                    elizakeyword.append(qi)
                    tt=handlePunc(say, qi)

                    tail = getTail(tt, qi)

                    replacedTail = tail.replace("我", "#")
                    replacedTail = replacedTail.replace("你", "我")
                    replacedTail = replacedTail.replace("#", "你")
                    tmpalist =aList[np.random.randint(len(aList))]
                    if tmpalist.find("*")>-1:
                      if len(replacedTail)<max_tail_length:
                        msg = [tail, tmpalist.replace("*", replacedTail)+"$"+qi+"$"]
                        patterns.append(msg)
                    else:
                      msg = [tail, tmpalist.replace("*", replacedTail)+"$"+qi+"$"]
                      patterns.append(msg)

        if patterns==[]:
            patterns.append([say, general_reply[np.random.randint(len(general_reply))].replace("*", say)+"$"+"None"+"$"])


        return getRandomPattern(patterns)

#             except:
#                 print(i)


    def getRandomPattern(patterns):
        return patterns[np.random.randint(len(patterns))]
#     def getTail(say, q):
#         print("lbk")
#         r= re.compile(r"(.*){}([^?.;]*)".format(q))
#         tmp = r.match(say)
#         print(tmp)
#         if tmp:
#             return tmp[1]
#         return ""
    def getTail(say, q):
        r= r"(.*)({})([^?.;]*)".format(q)
        tmp = re.findall(r,say)
        if tmp !=[] :
            return tmp[0][2]
        return ""

    def handlePunc(say, keyword):
        punct = [",", "\\\\.", "!", "-", "\\\\?", "，", "！", "？", ":", ";", "；", "：", "。", "、", "…"]
        tmppunc="".join(punct)
        post = say.find(keyword)
        if post == -1:
            return say
        r1=r"[{}\s](.*?{}.*?)[{}\s]".format(tmppunc,keyword,tmppunc)
        r2=r"[{}\s]*(.*?{}.*?)[{}\s]".format(tmppunc,keyword,tmppunc)
        r3=r"[{}\s](.*?{}.*?)[{}\s]*".format(tmppunc,keyword,tmppunc)
        if re.findall(r1,say) !=[]:

            return re.findall(r1,say)[0]
        elif re.findall(r2,say) !=[]:

            return re.findall(r2,say)[0]
        elif re.findall(r3,say) !=[]:

            return re.findall(r3,say)[0]
        else:
            return say
    reply = answer(aa,aa)
    return reply

def chatbot(chatbot_params):
  params_df=pd.DataFrame(chatbot_params).T
  mode = params_df[params_df.index=="mode"].order.item()
  params_df = params_df[params_df.index!="mode"].sort_values(by=["order"])
  while True:
    text=input("input:")
    label=0
    image = plt.imread('flyingPing.jpg')
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    for index, row in params_df.iterrows():
      if index == "general":
          if mode =="debug":
              print("chatbot: {}".format(general(text)))
          else:
              generaltext=general(text).split("$")
              print("chatbot: {}".format(generaltext[0]))
      elif index=="advice":
        modelpath = "/content/CantoneseChatbot/pretrain-model/regression_advice/bestmodel"
        advicepath= "/content/CantoneseChatbot/candidate/adviceall.csv"
        reply, score = regressionReply(text,modelpath,advicepath)
        if mode =="debug":
            print("chatbot: {}\nscore:{}".format(reply,score))
            if score > row["Threshold"]:
              break
            else:
              continue
        elif score > row["Threshold"]:
            print("chatbot: {}".format(reply))
            break
      elif index=="question":
        modelpath = "/content/CantoneseChatbot/pretrain-model/regression_question/bestmodel"
        advicepath= "/content/CantoneseChatbot/candidate/question.csv"
        reply, score = regressionReply(text,modelpath,advicepath)
        if mode =="debug":
            print("chatbot: {}\nscore:{}".format(reply,score))
            if score > row["Threshold"]:
              break
            else:
              continue
        elif score > row["Threshold"]:
            print("chatbot: {}".format(reply))
            break
      elif index=="restatement":
        modelpath = "/content/CantoneseChatbot/pretrain-model/regression_restatement/bestmodel"
        advicepath= "/content/CantoneseChatbot/candidate/restatement.csv"
        reply, score = regressionReply(text,modelpath,advicepath)
        if mode =="debug":
            print("chatbot: {}\nscore:{}".format(reply,score))
            if score > row["Threshold"]:
              break
            else:
              continue
        elif score > row["Threshold"]:
            print("chatbot: {}".format(reply))
            break
      elif index=="bertsum":
        print("chatbot: {}".format(general(text)))
        break
