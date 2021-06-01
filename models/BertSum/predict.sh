export CUDA_VISIBLE_DEVICES=0,1

python predict.py \
    --model_path output_span_6k/model_02-22-10:53:06/BertAbsSum_19.bin\
    --config_path output_span_6k/model_02-22-10:53:06/config.json\
    --eval_path data/BTS_span_data_6k/eval.csv\
    --bert_model pretrained_model/bert-base-chinese\
    --result_path result_span_6k
