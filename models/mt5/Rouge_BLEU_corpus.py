import pandas as pd
import numpy as np
from rouge_metric import PyRouge

from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bscore




def Rouge_BLEU_multiple(all_data,reference_Data=pd.DataFrame()):
    reference=[]
    hypothesis=[]
    reference_BLEU=[]
    hypothesis_BLEU=[]
    post_list =all_data.post.drop_duplicates().to_list()
    not_match_count=0
    for index, po in tqdm(enumerate(post_list)):
# Rouge
        tmp_df = all_data[all_data.post==po]
        if len(reference_Data)==0:
            tmp_refer=tmp_df.ref.to_list()
        else:
            tmp_refer = reference_Data[reference_Data.post==po].reference.to_list()
        if tmp_refer == []:
            not_match_count+=1
            tmp_refer=tmp_df.ref.to_list()
            
        tmp_hyp=tmp_df.hyp.to_list()
        tmp_a = [" ".join(list(x)) for x in tmp_refer]
        reference.append(tmp_a)
        tmp_b = [list(x) for x in tmp_refer]
        reference_BLEU.append(tmp_b)
        tmp_pre = [" ".join(list(tmp_hyp[0])) ]
        hypothesis=hypothesis+tmp_pre
        tmp_pre_BLEU =[list(tmp_hyp[0])] 
        hypothesis_BLEU=hypothesis_BLEU+tmp_pre_BLEU
    print("Total"+str(len(post_list)))
    print("not Match number"+str(not_match_count))
# BLEU
    


        
    rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, multi_ref_mode="best")
    score_rouge = rouge.evaluate(hypothesis, reference)
    print("Rouge")
    print(score_rouge)
    # bleu_score=[]
    # score = corpus_bleu(reference_BLEU, hypothesis_BLEU,weights=(1,0,0,0))
    # print("BLEU")
    # print(score)
# METEOR
    # meteor_score_list=[]
    # for index, hyp in enumerate(hypothesis):
    #     score  = meteor_score(reference[index],hyp)
    #     meteor_score_list.append(score)
    # print("METEOR")
    # print(np.mean(meteor_score_list))
# BERT SCORE
    # P, R, F1 = bscore(hypothesis, reference, lang="zh", verbose=True, rescale_with_baseline=True)
    # print("BERT_SCORE")
    # print(F1.mean())
    return score_rouge["rouge-l"]["f"]


def Rouge_BLEU_single(all_data):
    reference=[]
    hypothesis=[]
    reference_BLEU=[]
    hypothesis_BLEU=[]
    post_list =all_data.post.drop_duplicates().to_list()
    for index, po in enumerate(post_list):
# Rouge
        tmp_df = all_data[all_data.post==po]
        tmp_refer = tmp_df.ref.to_list()[:1]
        tmp_hyp=tmp_df.hyp.to_list()[:1]
        tmp_a = [" ".join(list(x)) for x in tmp_refer]
        reference.append(tmp_a)
        tmp_b = [list(x) for x in tmp_refer]
        reference_BLEU.append(tmp_b)
        tmp_pre = [" ".join(list(tmp_hyp[0])) ]
        hypothesis=hypothesis+tmp_pre
        tmp_pre_BLEU =[list(tmp_hyp[0])] 
        hypothesis_BLEU=hypothesis_BLEU+tmp_pre_BLEU
# BLEU
    


        
    scores = evaluator.get_scores(hypothesis, reference)
    print(scores)
    bleu_score=[]
    for index in range(len(hypothesis_BLEU)):
        re = reference_BLEU[index]
        ca = hypothesis_BLEU[index]

        score = sentence_bleu(re, ca, weights=(1, 0, 0, 0))


        bleu_score.append(score)
    print(np.mean(bleu_score))