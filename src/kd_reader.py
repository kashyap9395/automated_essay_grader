
import os 
from nltk import word_tokenize 
from nltk import pos_tag 
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP

# Reading essays into a list of strings 
def read_essays(path,fl):
	#directory = file_path
	essays_list = []
	for filename in fl:
	    if filename.endswith(".txt"): 
	        file_path = os.path.join(path, filename)
	        with open(file_path, 'r') as myfile:
	            essays_list.append(myfile.read().replace('\n', '').replace('\t','').replace('.', '. ').strip())
	        continue
	    else:
	        continue
# -*- coding: utf-8 -*-
	return essays_list 

# tokenizing essays into list of words and pos tags
def gen_word_pos(essays_list):
	essay_words = []
	essay_pos = []
	for essay in essays_list:
	    words = word_tokenize(essay)
	    essay_words.append(words)
	    essay_pos.append(pos_tag(words))

	return essay_words, essay_pos  

# tokenizing essays into list of sentences
def gen_sent(essays_list):
	essay_tokenized = []
	for i,essay in enumerate(essays_list):
	    temp = sent_tokenize(essay)
	    for j,sent in enumerate(temp): 
	        temp2 = word_tokenize(sent)
	        temp3 = []
	        # parser gives error for long sentences/incorrectly tokenized sentences
	        # this is a very rare case
	        if len(temp2) > 100:
	            tmp = len(temp2) // 30
	            for i in range(tmp):
	                s = 30*i
	                e = 30*(i+1)
	                
	                if i == tmp-1:
	                    e = len(temp2)
	                
	                temp3.append(" ".join(temp2[s:e]))
	            #temp3 = temp[j].replace("I", "|I").split("|")
	            temp4 = temp[:j]
	            temp4.extend(temp3)
	            if j < len(temp)-1:
	                temp4.extend(temp[j+1:])
	            temp = temp4
	            
	    essay_tokenized.append(temp)

	return essay_tokenized


# parsing each sentence in essay list
def gen_parse(essay_tokenized, stanford_path):
	nlp = StanfordCoreNLP(stanford_path)
	essay_parsed = []
	for i,essay in enumerate(essay_tokenized):
	    sent_parse = []
	    for j,sent in enumerate(essay):
	        sent_parse.append(nlp.parse(sent))
	    essay_parsed.append(sent_parse)
	nlp.close()

	return essay_parsed