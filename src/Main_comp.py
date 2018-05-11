
# -*- coding: utf-8 -*-
# main python file - call and grade essays here
# all imports here
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd 
from Essay import *
import kd_reader
import Length 
import c_1
import c_3
import d_1 
from progressbar import ProgressBar
import csv
import sys
from nltk.corpus import wordnet as wn 
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

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
F inal Score = 2 ∗ a − b + c.i + c.ii + 2 ∗ c.iii + 2 ∗ d.i [+3 ∗ d.ii]

Project 1 requirements : a,b,c1,c2 scores and considerations for the formula.
"""

# Initialize the progress bar here
pbar = ProgressBar()

# load the index file into a pandas frame to load the prompt and grades in the essay object
index_frame = pd.read_csv("../input/index.csv",delimiter=';',index_col = False)
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
                results_dict[files] = [a,b,c1,c2,c3,d1,d2[0],d2[1],d2[2],final_score,grade]




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
    #print("Results saved in results.csv file!")

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
                results_dict[files] = [a,b,c1,c2,c3,d1,d2[0],d2[1],d2[2],final_score,grade]




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
       
       
    #print("Results saved in results.csv file!")

print("Calculating A,C1,C3,D1")
essays_list = kd_reader.read_essays(data_path_train,train_files)
essay_words, essay_pos = kd_reader.gen_word_pos(essays_list)
essay_tokenized = kd_reader.gen_sent(essays_list)
essay_parsed = kd_reader.gen_parse(essay_tokenized,stanford_path)
length_train = Length.count_sent(essay_pos, essay_tokenized) #no. of sentences a 
c_1_count_train = c_1.count_c1(essay_parsed) # number of subj-verb agreement mistakes in each essay 
#part 2 code
c_3_count_train =  c_3.count_c3(essay_parsed)
sent_pos_train = d_1.gen_sent_pos(essay_tokenized)
d_1_count_train = d_1.count_d1(sent_pos_train,len(essay_parsed)) 
 


essays_list = kd_reader.read_essays(data_path_test,test_files) 
essay_words, essay_pos = kd_reader.gen_word_pos(essays_list)
essay_tokenized = kd_reader.gen_sent(essays_list)
essay_parsed = kd_reader.gen_parse(essay_tokenized,stanford_path)
length_test = Length.count_sent(essay_pos, essay_tokenized) #no. of sentences a 
c_1_count_test = c_1.count_c1(essay_parsed)
#part 2 code
c_3_count_test =  c_3.count_c3(essay_parsed)
sent_pos_test = d_1.gen_sent_pos(essay_tokenized)
d_1_count_test = d_1.count_d1(sent_pos_test,len(essay_parsed)) 


#print(c_1_count_test,c_1_count_train,length_test,length_train)
df_train = pd.DataFrame()
df_train['length_train'] = length_train
df_train['c1_train'] = c_1_count_train
df_train['c3_train'] = c_3_count_train
df_train['d1_train'] = d_1_count_train

df_test = pd.DataFrame()
df_test['length_test'] = length_test
df_test['c1_test'] = c_1_count_test
df_test['c3_test'] = c_3_count_test
df_test['d1_test'] = d_1_count_test

df_train.to_csv("../dump/trainvalues.csv")
df_test.to_csv("../dump/testvalues.csv") 
    

#------------------------------get output file------------------------------#
train = pd.read_csv("../dump/results.csv",names=["Filename","A","B","C1","C2","C3","D1","D2a","D2b","D2c","Final_Score","Grade","Cohort"],index_col=False)
test = pd.read_csv("../dump/results_test.csv",names=["Filename","A","B","C1","C2","C3","D1","D2a","D2b","D2c","Final_Score","Grade","Cohort"],index_col=False)

train.replace({"high":1,"low":0},inplace=True)
test.replace({"high":1,"low":0},inplace=True)

train_values = pd.read_csv("../dump/trainvalues.csv")
test_values = pd.read_csv("../dump/testvalues.csv") 

train['A'] = train_values['length_train']
train['C1'] = train_values['c1_train']
train['C3'] = train_values['c3_train']
train['D1'] = train_values['d1_train']

test['A'] = test_values['length_test']
test['C1'] = test_values['c1_test']
test['C3'] = test_values['c3_test']
test['D1'] = test_values['d1_test']

full_df = pd.concat([train,test])

mms = MinMaxScaler()

#full_df[["A","B","C1","C2","C3","D1","D2"]] = mms.fit_transform(full_df[["A","B","C1","C2","C3","D1","D2"]])

def scale(full_df):
        
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
        
        if 0.8<=row['C3']<1.01:
            full_df.set_value(i,'C3',5)
        elif 0.6<=row['C3']<0.8:
            full_df.set_value(i,'C3',4)
        elif 0.4<=row['C3']<0.6:
            full_df.set_value(i,'C3',3)
        elif 0.2<=row['C3']<0.4:
            full_df.set_value(i,'C3',2)
        elif row['C3']<0.2:
            full_df.set_value(i,'C3',1)

        if 0.8<=row['D1']<1.01:
            full_df.set_value(i,'D1',5)
        elif 0.6<=row['D1']<0.8:
            full_df.set_value(i,'D1',4)
        elif 0.4<=row['D1']<0.6:
            full_df.set_value(i,'D1',3)
        elif 0.2<=row['D1']<0.4:
            full_df.set_value(i,'D1',2)
        elif row['D1']<0.2:
            full_df.set_value(i,'D1',1)

        if 0.8<=row['D2']<1.01:
            full_df.set_value(i,'D2',5)
        elif 0.6<=row['D2']<0.8:
            full_df.set_value(i,'D2',4)
        elif 0.4<=row['D2']<0.6:
            full_df.set_value(i,'D2',3)
        elif 0.2<=row['D2']<0.4:
            full_df.set_value(i,'D2',2)
        elif row['D2']<0.2:
            full_df.set_value(i,'D2',1)
    return full_df
train_trans = train.copy()


test_trans = test.copy()
#train_trans['D2'] = (train['D2a'] + train['D2b'] + train['D2c'])/3.0
#test_trans['D2'] = (test['D2a'] + test['D2b'] + test['D2c'])/3.0

#train_trans.drop(['D2a','D2b','D2c'], axis = 1, inplace=True)
#test_trans.drop(['D2a','D2b','D2c'], axis = 1, inplace=True)

#print(test_trans)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
#x_train = pd.DataFrame()
#x_train[["A","B","C1","C2","C3","D1","D2a","D2b","D2c"]] = mms.fit_transform(train_trans[['A','B','C1','C2',"D2a","D2b","D2c"]])
#x_train['D2'] = 0
#x_train['D2_temp'] = x_train['D2a'].astype(float) + x_train['D2b'].astype(float) + x_train['D2c'].astype(float)
#x_train['D2'] = x_train['D2_temp'].apply(lambda x:x/3)
#x_train.drop(['D2a','D2b','D2c','D2_temp'], axis = 1, inplace=True)

x_train = mms.fit_transform(train[['A','B','C1','C2','C3','D1',"D2a","D2b","D2c"]])
#temp = (x_train[:,-1] + x_train[:,-2] + x_train[:,-3])/3.0
#print(x_train)
#x_train = np.delete(x_train, np.s_[3:6], 1)
#print(x_train)
#x_train = np.c_[x_train,temp]
#print(x_train)


y_train = train[['Grade']].values.ravel()

#x_test = pd.DataFrame()
#x_test[["A","B","C1","C2","C3","D1","D2a","D2b","D2c"]] = mms.fit_transform(test_trans[['A','B','C1','C2',"D2a","D2b","D2c"]])
#x_test['D2'] = 0
#x_test['D2_temp'] = x_test['D2a'].astype(float) + x_test['D2b'].astype(float) + x_test['D2c'].astype(float)
#x_test['D2'] = x_test['D2_temp'].apply(lambda x:x/3)
#x_test.drop(['D2a','D2b','D2c','D2_temp'], axis = 1, inplace=True)

x_test = mms.fit_transform(test[['A','B','C1','C2','C3','D1',"D2a","D2b","D2c"]])
#temp = (x_test[:,-1] + x_test[:,-2] + x_test[:,-3])/3.0
#print(x_train)
#x_test = np.delete(x_test, np.s_[3:6], 1)
#print(x_train)
#x_test = np.c_[x_test,temp]
#print(x_train)


y_test = test[['Grade']].values.ravel()

logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
print("------------------Accuracy----------------------")
#print(logreg.score(x_test,y_test)*100)
print(accuracy_score(y_test,y_pred))
print("---------------------------------------------")
print(classification_report(y_pred,y_test))

train_trans[["A","B","C1","C2","C3","D1","D2a","D2b","D2c"]] = mms.fit_transform(train[["A","B","C1","C2","C3","D1","D2a","D2b","D2c"]]) 
train_trans['D2_temp'] = train_trans['D2a'].astype(float) + train_trans['D2b'].astype(float) + train_trans['D2c'].astype(float)
train_trans.insert(5, "D2", train_trans['D2_temp'].apply(lambda x:x/3))
train_trans.drop(['D2a','D2b','D2c','D2_temp'], axis = 1, inplace=True)
train_trans = scale(train_trans)

test_trans[["A","B","C1","C2","C3","D1","D2a","D2b","D2c"]] = mms.fit_transform(test[["A","B","C1","C2","C3","D1","D2a","D2b","D2c"]]) 
test_trans['D2_temp'] = test_trans['D2a'].astype(float) + test_trans['D2b'].astype(float) + test_trans['D2c'].astype(float)
#test_trans['D2'] = test_trans['D2_temp'].apply(lambda x:x/3)
test_trans.insert(5, "D2", test_trans['D2_temp'].apply(lambda x:x/3))
test_trans.drop(['D2a','D2b','D2c','D2_temp'], axis = 1, inplace=True)
test_trans = scale(test_trans)

# dump results into folder
dump_results = test_trans
test_trans['Grade'] = y_pred
#test_trans['C3'] = 0
#test_trans['D1'] = 0
#test_trans['D2'] = 0
test_trans['Final_Score'] = (2*test_trans['A'])-test_trans['B']+test_trans['C1']+test_trans['C2']+test_trans['C3']+test_trans['D1']+test_trans['D2']
test_trans.drop(['Cohort'], axis=1,inplace=True)
dump_results.to_csv("../output/results_with_grade.txt",index=False)     
        
    





