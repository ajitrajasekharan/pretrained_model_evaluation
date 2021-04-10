import torch
from transformers import *
import pdb
import operator
from collections import OrderedDict
import sys
import traceback
import argparse
import string
import numpy as np
import json
import CosineCacher as CC


import logging

#DEFAULT_MODEL_PATH='bert-large-cased'
DEFAULT_MODEL_PATH='./'
DEFAULT_TO_LOWER=False
DEFAULT_INPUT_FILE='normalized_test.csv'
DEFAULT_EMBEDS_FILE="bert_vectors.txt"
DEFAULT_LABELS_FILE="map_labels.txt"
DEFAULT_OUTPUT_FILE="results.tsv"
DEFAULT_VOCAB_FILE="vocab.txt"
TOP_K_RANK=0.001
DEFAULT_GRAPH_SIZE = 5
BIAS_THRESH = 0
MAX_E_COMPARE = 5

def init_model(model_path,to_lower):
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()
    return model,tokenizer



def get_info(model,tokenizer,text,is_cls,cluster_size):
    text = '[CLS] ' + text + ' [SEP]' 
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    masked_index = 0
    if (not is_cls):
            for i in range(len(tokenized_text)):
                if (tokenized_text[i] == "entity"):
                    masked_index = i
                    tokenized_text[masked_index] = "[MASK]"
                    indexed_tokens[masked_index] = 103
                    break

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    print(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
        tfidf_equiv  = predictions[0][2]
        results_dict = {}
        for j in range(len(predictions[0][0][0,masked_index])):
            tok = tokenizer.convert_ids_to_tokens([j])[0]
            if (tfidf_equiv[j] <= BIAS_THRESH):
                results_dict[tok] = float(predictions[0][0][0,masked_index][j].tolist())
        sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))
        rank = 1
        ret_arr = []
        for word in sorted_d:
            if (word in string.punctuation or word.startswith('##') or len(word) == 1 or word.startswith('.') or word.startswith('[')):
                continue
            ret_arr.append(word)
            if (rank >= cluster_size):
                break
            rank += 1
        return ' '.join(ret_arr).lstrip()


#TBD. To be taken from config
map_actual = {"autoimmune_disease":"DIS","cancer":"DIS","cardiovascular_disease":"DIS","cell":"BIO","disease":"DIS","drug":"DRUG","tissue":"BIO"}

def pick_entity(actual,inp_dict):
     for e in inp_dict:
        if (e != "OTHER"):
            return e
        else:
            continue
     return "OTHER"

def update_cf_matrix(confusion_matrix,predicted_dict,actual):
    actual = map_actual[actual]
    if (actual not in confusion_matrix):
            confusion_matrix[actual] = OrderedDict()
    row = confusion_matrix[actual]
    predicted = pick_entity(actual,predicted_dict)
    if (predicted not in row):
        row[predicted] = 1
    else:
        row[predicted] += 1

def compute_f1_score(in_confusion_matrix):
    #Pick all unique entities
    max_entities = {}
    for actual in in_confusion_matrix:
        if (actual not in max_entities):
            max_entities[actual] = 0
        actual_dict = in_confusion_matrix[actual]
        for col in actual_dict:
            if (col not in max_entities):
                max_entities[col] = 0


    #Construct a square matrix
    confusion_matrix = OrderedDict()
    for row in max_entities:
        confusion_matrix[row] = OrderedDict()
        row_dict = confusion_matrix[row]
        for col in max_entities:
            if (row in in_confusion_matrix and col in in_confusion_matrix[row]):
                confusion_matrix[row][col] = in_confusion_matrix[row][col]
            else:
                confusion_matrix[row][col] = 0


    #Precision compute. Column major traversal
    precision_dict = OrderedDict()
    main_index = 0
    for curr_col in max_entities:
        numerator = 0
        denominator = 0
        assert(curr_col in confusion_matrix)
        aux_index = 0
        for actual in confusion_matrix:
            if (actual != curr_col):
                aux_index += 1
                continue
            actual_dict = confusion_matrix[actual]
            if (aux_index == main_index):
                numerator = actual_dict[curr_col]
            denominator += actual_dict[curr_col]
            aux_index += 1
        if (denominator == 0):
            precision_val = 0
        else:
            precision_val = round(float(numerator)/denominator,2)
        precision_dict[curr_col] = precision_val
        main_index += 1
       
    #Recall compute. Row major traversal
    recall_dict = OrderedDict()
    for actual in confusion_matrix:
        actual_dict = confusion_matrix[actual]
        numerator = 0
        denominator = 0
        for predicted in actual_dict:
            if (predicted == actual):
                    numerator = actual_dict[predicted]
            denominator += actual_dict[predicted]
        if (denominator == 0):
            recall_val = 0
        else:
            recall_val = round(float(numerator)/denominator,2)
        recall_dict[actual] = recall_val
              
    f1_scores = OrderedDict()
    count = 0
    cull_score = 0
    full_score = 0
    max_score = 0
    for term in recall_dict:
        if(term not in precision_dict):
            curr_score = 0    
        else:
                if (precision_dict[term] + recall_dict[term] == 0):
                    curr_score = 0
                else:
                    curr_score = 2*(precision_dict[term]  *recall_dict[term])/(precision_dict[term] + recall_dict[term])
        f1_scores[term] = round(curr_score,2)
        full_score += curr_score
        if (curr_score > max_score):
            max_score = curr_score
        count += 1
    ret_dict = {"matrix":confusion_matrix,"precision":precision_dict,"recall":recall_dict,"f1_score":f1_scores,"max_f1":round(max_score,2)}
    return ret_dict
    
            
            
        

def batch_process(results,model,tokenizer,cosine_cacher):
    inp_file = results.input
    output_file = results.output_file
    cluster_size = results.cluster_size
    
    cache_cls_prediction_dict = {}
    ofp = open(output_file,"w")
    heading = "#CLS_PREDICTION_ENTITY|MASK_PREDICTION_ENTITY|ACTUAL_ENTITY|CLS_PREDICTIONS|MASKED_PREDICTIONS|MASKED_WORD|SENTENCE\n"
    print(heading)
    ofp.write(heading)
    count = 1
    cls_confusion_matrix = OrderedDict() 
    masked_confusion_matrix = OrderedDict()
    with open(inp_file) as fp:
        for line in fp:
            line = line.lower().rstrip('\n')
            fields = line.split('|')
            if (fields[0] not in cache_cls_prediction_dict):
                cls_words = get_info(model,tokenizer,fields[0],True,cluster_size)
                assert(len(cls_words) > 0) 
                _,cls_mean,cls_std,sorted_cls_d = cosine_cacher.find_pivot_subgraph(cls_words.split())
                assert(len(sorted_cls_d) >  0)
                cls_entity_types = cosine_cacher.get_entity_type_for_terms(sorted_cls_d)
                update_cf_matrix(cls_confusion_matrix,cls_entity_types,fields[1])
                cache_cls_prediction_dict[fields[0]] = {"entity":cls_entity_types,"actual":fields[1],"words":cls_words}
            else:
                cls_entity_types = cache_cls_prediction_dict[fields[0]]["entity"] 
                cls_words = cache_cls_prediction_dict[fields[0]]["words"] 
                update_cf_matrix(cls_confusion_matrix,cls_entity_types,fields[1])
            masked_words = get_info(model,tokenizer,fields[2],False,cluster_size)
            _,masked_mean,masked_std,sorted_masked_d = cosine_cacher.find_pivot_subgraph(masked_words.split())
            masked_entity_types = cosine_cacher.get_entity_type_for_terms(sorted_masked_d)
            update_cf_matrix(masked_confusion_matrix,masked_entity_types,fields[1])
            out_str = "{}|{}|{}|{}|{}|{}|{}".format(cls_entity_types,masked_entity_types,fields[1],cls_words,masked_words,fields[0],fields[2])
            print(str(count),']',out_str)
            ofp.write(out_str + '\n')
            count += 1  
    masked_f1_score = compute_f1_score(masked_confusion_matrix)
    cls_f1_score = compute_f1_score(cls_confusion_matrix)
    out_str = "#Masked F1-score:{:.2f}|CLS F1-score:{:.2f}|Masked Confusion matrix:{}|CLS confusion matrix:{}".format(masked_f1_score["max_f1"],cls_f1_score["max_f1"],str(masked_f1_score),str(cls_f1_score))
    print(out_str)
    ofp.write(out_str + '\n')
    ofp.flush()
    ofp.close()

def bucket_vals(arr):
        bucket = {}
        for val in arr:
                val = round(val,4)
                if (val in bucket):
                    bucket[val] += 1
                else:
                    bucket[val] = 1
        sorted_d = OrderedDict(sorted(bucket.items(), key=lambda kv: kv[0], reverse=True))
        return sorted_d
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting neighbors to a word in sentence using BERTMaskedLM. Neighbors are from BERT vocab (which includes subwords and full words). Input needs to be in the format term1|CLASS|sentence with the word entity where the maskin of term1(which could be a phrase) was done ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-tolower', action="store", dest="tolower", default=DEFAULT_TO_LOWER,type=bool,help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.add_argument('-input', action="store", dest="input", default=DEFAULT_INPUT_FILE,help='Input file with sentences')
    parser.add_argument('-embeds_file', action="store", dest="embeds_file", default=DEFAULT_EMBEDS_FILE,help='Embedding file')
    parser.add_argument('-output_file', action="store", dest="output_file", default=DEFAULT_OUTPUT_FILE,help='Output results file')
    parser.add_argument('-vocab_file', action="store", dest="vocab_file", default=DEFAULT_VOCAB_FILE,help='Vocab file')
    parser.add_argument('-cluster_size', action="store", dest="cluster_size", default=DEFAULT_GRAPH_SIZE,type=int,help='number of results to pick')
    parser.add_argument('-labels_file', action="store", dest="labels_file", default=DEFAULT_LABELS_FILE,help='labels file with entity cluster pivots')

    results = parser.parse_args()
    try:
        model,tokenizer = init_model(results.model,results.tolower)
        cosine_cacher = CC.CosineCacher(tokenizer,results.vocab_file,results.embeds_file,results.labels_file)
        batch_process(results,model,tokenizer,cosine_cacher)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
