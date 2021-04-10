import pdb
import sys
import operator
from collections import OrderedDict
import subprocess
import numpy as  np
import json
import math
from transformers import *
import sys
import logging


BERT_TERMS_START=106
UNK_ID = 100
#Original setting for cluster generation. 

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')



def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_dict = json.loads(fp.read())
    return embeds_dict


def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file) as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[term] = count
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict

def read_labels(labels_file):
    terms_dict = OrderedDict()
    print("Reading: ",labels_file,"...")
    with open(labels_file) as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            term = term.split()
            if (len(term) >= 6):
                labels_arr = term[0].split('/')
                rest_arr = ' '.join(term[6:])[1:-1].replace('\'','').replace(' ','').split(',')
                for label in labels_arr:
                        for val in rest_arr:
                            if (val not in terms_dict):
                                terms_dict[val] = {}
                            if (label not in terms_dict[val]):
                                terms_dict[val][label] = 1
                count += 1
            else:
                print("Invalid line:",term)
                pdb.set_trace()
                assert(0)
    print("count of labels in " + labels_file + ":", len(terms_dict))
    return terms_dict

def is_filtered_term(key): #Words selector. skiping all unused and special tokens
        return True if (str(key).startswith('#') or str(key).startswith('[')) else False

def init_model(model_path,to_lower):
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
    model = BertForMaskedLM.from_pretrained(model_path)
    #tokenizer = RobertaTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
    #model = RobertaForMaskedLM.from_pretrained(model_path)
    model.eval()
    return model,tokenizer

class CosineCacher:
    def __init__(self, tokenizer,terms_file,embeds_file,labels_file):
        self.tokenizer = tokenizer
        self.terms_dict = read_terms(terms_file)
        self.embeddings = read_embeddings(embeds_file)
        self.labels_dict = read_labels(labels_file)
        self.embeds_cache = {}
        self.cosine_cache = {}
        self.cache = True
        self.dist_threshold_cache = {}

    def get_embedding(self,text):
        if (self.cache and text in self.embeds_cache):
            return self.embeds_cache[text]
        tokenized_text = text.split()
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        #print(text,indexed_tokens)
        vec =  self.get_vector(indexed_tokens)
        if (self.cache):
                self.embeds_cache[text] = vec
        return vec


    def get_vector(self,indexed_tokens):
        vec = None
        if (len(indexed_tokens) == 0):
            return vec
        #pdb.set_trace()
        for i in range(len(indexed_tokens)):
            term_vec = self.embeddings[indexed_tokens[i]]
            if (vec is None):
                vec = np.zeros(len(term_vec))
            vec += term_vec
        sq_sum = 0
        for i in range(len(vec)):
            sq_sum += vec[i]*vec[i]
        sq_sum = math.sqrt(sq_sum)
        for i in range(len(vec)):
            vec[i] = vec[i]/sq_sum
        #sq_sum = 0
        #for i in range(len(vec)):
        #    sq_sum += vec[i]*vec[i]
        return vec

    def calc_inner_prod(self,text1,text2):
        if (self.cache and text1 in self.cosine_cache and text2 in self.cosine_cache[text1]):
            return self.cosine_cache[text1][text2]
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)
        if (vec1 is None or vec2 is None):
            #print("Warning: at least one of the vectors is None for terms",text1,text2)
            return 0
        val = np.inner(vec1,vec2)
        if (self.cache):
            if (text1 not in self.cosine_cache):
                self.cosine_cache[text1] = {}
            self.cosine_cache[text1][text2] = val
        return val

    def get_entity_type_for_terms(self,pivots_arr):
        count = 0
        top_labels = {}
        for pivot in pivots_arr:
            if (pivot not in self.labels_dict):
                top_labels_dict = {"OTHER":1}
            else:
                top_labels_dict = self.labels_dict[pivot]
            for top_label in top_labels_dict:
                if (top_label not in top_labels):
                    top_labels[top_label] = 1
                else:
                    top_labels[top_label] += 1
                count += 1
        sorted_d = OrderedDict(sorted(top_labels.items(), key=lambda kv: kv[1], reverse=True))
        return sorted_d
            

    def get_terms_above_threshold(self,term1,threshold):
        final_dict = {}
        for k in self.terms_dict:
            term2 = k.strip("\n")
            val = self.calc_inner_prod(term1,term2)
            val = round(val,2)
            if (val > threshold):
                final_dict[term2] = val
        sorted_d = OrderedDict(sorted(final_dict.items(), key=lambda kv: kv[1], reverse=True))
        return sorted_d

    def print_terms_above_threshold(self,term1,threshold):
        fp = open("above_t.txt","w")
        sorted_d = self.get_terms_above_threshold(term1,threshold)
        for k in sorted_d:
                print(k," ",sorted_d[k])
                fp.write(str(k) + " " + str(sorted_d[k]) + "\n")
        fp.close()

    #given n terms, find the mean of the connection strengths of subgraphs considering each term as pivot.
    #return the mean of max strength term subgraph
    def find_pivot_subgraph(self,terms):
        max_mean = 0
        std_dev = 0
        max_mean_term = None
        means_dict = {}
        if (len(terms) == 1):
            return terms[0],1,0,{terms[0]:1}
        for i in terms:
            full_score = 0
            count = 0
            full_dict = {}
            for j in terms:
                if (i != j):
                    val = self.calc_inner_prod(i,j)
                    #print(i+"-"+j,val)
                    full_score += val
                    full_dict[count] = val
                    count += 1
            if (len(full_dict) > 0):
                mean  =  float(full_score)/len(full_dict)
                means_dict[i] = mean
                #print(i,mean)
                if (mean > max_mean):
                    #print("MAX MEAN:",i)
                    max_mean_term = i
                    max_mean = mean
                    std_dev = 0
                    for k in full_dict:
                        std_dev +=  (full_dict[k] - mean)*(full_dict[k] - mean)
                    std_dev = math.sqrt(std_dev/len(full_dict))
                    #print("MEAN:",i,mean,std_dev)
        #print("MAX MEAN TERM:",max_mean_term)
        sorted_d = OrderedDict(sorted(means_dict.items(), key=lambda kv: kv[1], reverse=True))
        return max_mean_term,round(max_mean,2),round(std_dev,2),sorted_d





def get_words():
    while (True):
        print("Enter words separated by spaces : q to quit")
        sent = input()
        #print(sent)
        if (sent == "q"):
            print("Exitting")
            sys.exit(1)
        if (len(sent) > 0):
            break
    return sent.split()


def graph_test(b_embeds):
    while (True):
        words = get_words()
        max_mean_term,max_mean, std_dev,s_dict = b_embeds.find_pivot_subgraph(words)
        desc = ""
        for i in s_dict:
            desc += i + " "
        print("PSG score:",max_mean_term,max_mean, std_dev,s_dict)
        print(desc)

def main():
        model,tokenizer = init_model("./",False)
        cosine_cacher = CosineCacher(tokenizer,"vocab.txt","bert_vectors.txt")
        #cosine_cacher.print_terms_above_threshold("lithium",-1)
        #graph_test(cosine_cacher)
        
    

if __name__ == '__main__':
    main()
