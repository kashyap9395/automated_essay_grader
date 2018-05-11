# -*- coding: utf-8 -*-
# main python file - call and grade essays here
# all imports here 
import os
import pandas as pd 
from Essay import *
from Evaluate import *
import kd_reader
import Length 
import c_1
from progressbar import ProgressBar 
import csv
import sys
from nltk.corpus import wordnet as wn 
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import statsmodels.api as sm
stop_words = set(stopwords.words('english'))



"""
We need the following components to calculate the final score:
a. Length of the essay.
b. Spelling Mistakes.
c1. Subject verb agreement and person to number agreement.
c2. Verb tense / missing verb / extra verb -is verb tense used correctly? Is a verb missing, e.g. an
auxiliary? For example, in the example of low essay above, the sequence will be not agree
is incorrect. Normally the verb to be is not followed by another infinitival verb, but either a
participle or a progressive tense.
c3. Sentence formation -are the sentences formed properly? i.e. beginning and ending properly,
is the word order correct, are the constituents formed properly? are there missing words or
constituents (prepositions, subject, object etc.)?
d1. Is the essay coherent?
d2. Does the essay answer the question / address the topic?

Then we have the final score as:
Final Score = 2 ∗ a − b + c.i + c.ii + 2 ∗ c.iii + 2 ∗ d.i [+3 ∗ d.ii]

Project 1 requirements : a,b,c1,c2 scores and considerations for the formula.
"""

# Initialize the progress bar here
pbar = ProgressBar()

# load the index file into a pandas frame to load the prompt and grades in the essay object
index_frame = pd.read_csv("../input/training/index.csv",delimiter=';',index_col = False)
# remove trailing spaces in the prompt column
index_frame['prompt'] = index_frame['prompt'].str.strip()
# initialize the results dict
results_dict = {}
# Essay directory
data_path_train = "../input/training/"
data_path_test = "../input/testing/"


wordl = {}
dets = []

def save_tags(d):
    """
    saves lexicons from each essay into the grammar
        :param d: 
    """
    for key,value in d.items():
        if not key in wordl:
            wordl[key] = set()
        wordl[key] = wordl[key].union(set(value))
    



def get_grammar(path,files_list):
    """
    Takes file list as input to save lexicons in the directory
        :param files_list: 
    """
    for files in pbar(files_list):
        if files.endswith(".txt"):
            with open(path+files) as f_:
                text=f_.read().replace('\t', '').replace('\t','').replace('.', '. ').strip()
                
                prompt = index_frame.loc[index_frame['filename']==files,'prompt'].iloc[0]
                grade = index_frame.loc[index_frame['filename']==files,'grade'].iloc[0]
                
                essay = Essay(text,prompt,grade,stop_words)
                
                
                essay.set_words()
                save_tags(essay.get_tagged())
            
            
def dump():
    """
    stiches together a line to be dumped into the file
    """
    
    quotes = '"'
    for key,value in wordl.items():
        s = ""+key+" -> "
        token = " | ".join(quotes+c+quotes for c in list(value))
        s+=token
        dets.append(s+"\n")
    with open('../dump/tags.txt', 'w') as f:
        f.writelines(dets)

print("Generating Grammar...")
get_grammar(data_path_train,os.listdir(data_path_train))
pbar = ProgressBar()
get_grammar(data_path_test,os.listdir(data_path_test))
dump()
stanford_path = sys.argv[1]
train_files = os.listdir(data_path_train)
test_files = os.listdir(data_path_test)

#---------------------------- Save grammar to files for future reference------------------------------#
terminals=""
rules=""
with open('../dump/tags.txt', 'r') as myfile:
    for line in myfile:
        terminals = terminals+line
    
with open('rules.txt', 'r') as myfile:
    for line in myfile:
        rules = rules+line
    
grammar = rules+"\n"+terminals

print("Training...")
#------------------------------Get the raw values that are used to generate a rating-------------------#
pbar = ProgressBar()
# read all files in the directory and start processing
for files in pbar(os.listdir(data_path_train)):
    if files.endswith(".txt"):
        with open(data_path_train+files) as f_:
            text=f_.read().replace('\t', '').replace('\t','').replace('.', '. ').strip()
            
            prompt = index_frame.loc[index_frame['filename']==files,'prompt'].iloc[0]
            grade = index_frame.loc[index_frame['filename']==files,'grade'].iloc[0]
            
            essay = Essay(text,prompt,grade,stop_words)
            
            essay.set_words()
            #essay.get_tagged()
            essay.grammar = grammar
            #print(files)
            
            c1,a = essay.get_length()
            b = essay.get_spellingmistakes()
            #c1 = essay.get_sv_agreement()
            c2 = essay.get_verb_usage()
            c3 = essay.get_sentence_formation()
            d1 = essay.get_coherence()
            d2 = essay.get_topic_relevance()

            evaluate = Evaluate(a,b,c1,c2,c3,d1,d2)
            final_score  = evaluate.get_score()
            
            
            if not files in results_dict:
                results_dict[files] = [a,b,c1,c2,c3,d1,d2,final_score,grade]




# Dump raw values to csv
with open('../dump/results.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in results_dict.items():
        li = []
        li.append(key)
        for v in value:
            li.append(v)
        li.append(0)
        writer.writerow(li)
    print("Results saved in results.csv file!")

pbar = ProgressBar()
# read all files in the directory and start processing
results_dict = {}
for files in pbar(os.listdir(data_path_test)):
    if files.endswith(".txt"):
        with open(data_path_test+files) as f_:
            text=f_.read().replace('\t', '').replace('\t','').replace('.', '. ').strip()
            
            prompt = index_frame.loc[index_frame['filename']==files,'prompt'].iloc[0]
            grade = index_frame.loc[index_frame['filename']==files,'grade'].iloc[0]
            
            essay = Essay(text,prompt,grade,stop_words)
            
            essay.set_words()
            #essay.get_tagged()
            essay.grammar = grammar
            #print(files)
            
            c1,a = essay.get_length()
            b = essay.get_spellingmistakes()
            #c1 = essay.get_sv_agreement()
            c2 = essay.get_verb_usage()
            c3 = essay.get_sentence_formation()
            d1 = essay.get_coherence()
            d2 = essay.get_topic_relevance()

            evaluate = Evaluate(a,b,c1,c2,c3,d1,d2)
            final_score  = evaluate.get_score()
            
            
            if not files in results_dict:
                results_dict[files] = [a,b,c1,c2,c3,d1,d2,final_score,grade]




# Dump raw values to csv
with open('../dump/results_test.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in results_dict.items():
        li=[]
        li.append(key)
        for v in value:
            li.append(v)
        li.append(1)
        writer.writerow(li)
       
       
    print("Results saved in results.csv file!")

print("Calculating A and C1")
essays_list = kd_reader.read_essays(data_path_train,train_files)
essay_words, essay_pos = kd_reader.gen_word_pos(essays_list)
essay_tokenized = kd_reader.gen_sent(essays_list)
essay_parsed = kd_reader.gen_parse(essay_tokenized,stanford_path)
length_train = Length.count_sent(essay_pos, essay_tokenized) #no. of sentences a 
c_1_count_train = c_1.count_c1(essay_parsed) # number of subj-verb agreement mistakes in each essay  


essays_list = kd_reader.read_essays(data_path_test,test_files)
essay_words, essay_pos = kd_reader.gen_word_pos(essays_list)
essay_tokenized = kd_reader.gen_sent(essays_list)
essay_parsed = kd_reader.gen_parse(essay_tokenized,stanford_path)
length_test = Length.count_sent(essay_pos, essay_tokenized) #no. of sentences a 
c_1_count_test = c_1.count_c1(essay_parsed)

#print(c_1_count_test,c_1_count_train,length_test,length_train)
df_train = pd.DataFrame()
df_train['length_train'] = length_train
df_train['c1_train'] = c_1_count_train

df_test = pd.DataFrame()
df_test['length_test'] = length_test
df_test['c1_test'] = c_1_count_test

df_train.to_csv("../dump/trainvalues.csv")
df_test.to_csv("../dump/testvalues.csv") 
    

#------------------------------get output file------------------------------#
train = pd.read_csv("../dump/results.csv",names=["Filename","A","B","C1","C2","C3","D1","D2","Final_Score","Grade","Cohort"],index_col=False)
test = pd.read_csv("../dump/results_test.csv",names=["Filename","A","B","C1","C2","C3","D1","D2","Final_Score","Grade","Cohort"],index_col=False)

train.replace({"high":1,"low":0},inplace=True)
test.replace({"high":1,"low":0},inplace=True)

train_values = pd.read_csv("../dump/trainvalues.csv")
test_values = pd.read_csv("../dump/testvalues.csv") 

train['A'] = train_values['length_train']
train['C1'] = train_values['c1_train']

test['A'] = test_values['length_test']
test['C1'] = test_values['c1_test']

full_df = pd.concat([train,test])

mms = MinMaxScaler()

full_df[["A","B","C1","C2","C3","D1","D2"]] = mms.fit_transform(full_df[["A","B","C1","C2","C3","D1","D2"]])

for i, row in full_df.iterrows():
    if 0.8<=row['A']<1.01:
        full_df.set_value(i,'A',5)
    elif 0.6<=row['A']<0.8:
        full_df.set_value(i,'A',4)
    elif 0.4<=row['A']<0.6:
        full_df.set_value(i,'A',3)
    elif 0.2<=row['A']<0.4:
        full_df.set_value(i,'A',2)
    elif row['A']<0.2:
        full_df.set_value(i,'A',1)
    
    if 0.8<=row['B']<1.01:
        full_df.set_value(i,'B',4)
    elif 0.6<=row['B']<0.8:
        full_df.set_value(i,'B',3)
    elif 0.4<=row['B']<0.6:
        full_df.set_value(i,'B',2)
    elif 0.2<=row['B']<0.4:
        full_df.set_value(i,'B',1)
    elif row['B']<0.2:
        full_df.set_value(i,'B',0)
        
    if 0.8<=row['C1']<1.01:
        full_df.set_value(i,'C1',5)
    elif 0.6<=row['C1']<0.8:
        full_df.set_value(i,'C1',4)
    elif 0.4<=row['C1']<0.6:
        full_df.set_value(i,'C1',3)
    elif 0.2<=row['C1']<0.4:
        full_df.set_value(i,'C1',2)
    elif row['C1']<0.2:
        full_df.set_value(i,'C1',1)
    
    
    if 0.8<=row['C2']<1.01:
        full_df.set_value(i,'C2',5)
    elif 0.6<=row['C2']<0.8:
        full_df.set_value(i,'C2',4)
    elif 0.4<=row['C2']<0.6:
        full_df.set_value(i,'C2',3)
    elif 0.2<=row['C2']<0.4:
        full_df.set_value(i,'C2',2)
    elif row['C2']<0.2:
        full_df.set_value(i,'C2',1)
    
train['intercept'] = 1.0

model = sm.Logit(train.Grade, train[['intercept', "A","B","C1","C2"]])
result = model.fit()

print("#---------------------------------------------------------------#")
print(result.summary2())

# the odds
np.exp(result.params)
test = full_df.loc[full_df['Cohort'] == 1]

dump_df = test[["Filename","A","B","C1","C2"]].copy()
dump_df['Grade'] = 'Unknown'

dump_df.to_csv("../output/results.csv") 
    
        
    





