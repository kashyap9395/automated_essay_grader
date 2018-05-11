# -*- coding: utf-8 -*-
import kd_reader
import a 
import c_1
import c_3
import d_1
stanford_path = r'C:\Users\Kashyap\stanford-corenlp-full-2017-06-09' #path to stanford-corenlp-folder
file_path = r"essays_dataset\essays" # path to essays

essays_list = kd_reader.read_essays(file_path)
essay_words, essay_pos = kd_reader.gen_word_pos(essays_list)
essay_tokenized = kd_reader.gen_sent(essays_list)
essay_parsed = kd_reader.gen_parse(essay_tokenized,stanford_path)
length = a.count_sent(essay_pos, essay_tokenized) #no. of sentences a 
c_1_count = c_1.count_c1(essay_parsed) # number of subj-verb agreement mistakes in each essay 

#Recent additions 
print("Doing c3")
c_3_count = c_3.count_c3(essay_parsed)
sent_pos = d_1.gen_sent_pos(essay_tokenized)
print("Doing d1")
d_1_count = d_1.count_d1(sent_pos) 