# -*- coding: utf-8 -*-
import nltk
#nltk.download('names')
# Corpus which consists of male and female names dataset
from nltk.corpus import names
from nltk import word_tokenize 
from nltk import pos_tag 
import itertools

def gen_sent_pos(essay_tokenized):
	sent_pos = []
	for essay in essay_tokenized:
	    temp = []
	    for sent in essay:
	        temp.append(pos_tag(word_tokenize(sent)))
	        
	    sent_pos.append(temp)

	return sent_pos


# d-1
def count_d1(sent_pos,essay_count):
	d1_score = []
	male_names = [name.lower() for name in names.words("male.txt")]
	female_names = [name.lower() for name in names.words("female.txt")]
	male_pronouns = ['he', 'him', 'his', 'himself']
	female_pronouns = ['she', 'her', 'hers', 'herself']
	male_criteria = male_names + male_pronouns + ['person','student']
	female_criteria = female_names + female_pronouns + ['person','student']
	plural_third = ['they', 'their','them', 'themselves', 'theirs']
	criteria_word = plural_third + ['and']
	criteria_pos = ['NNS', 'NNPS']
	for i in range(essay_count):
	    mistake = 0
	    for j in range(len(sent_pos[i])):
	        for k,tag in enumerate(sent_pos[i][j]):
	            
	            
	            if tag[0] in plural_third:
	                window_pos = sent_pos[i][max(0,j-2):j]
	                window_pos.append([(t1,t2) for m,(t1,t2) in enumerate(sent_pos[i][j]) if m < k])
	                combined_window = list(itertools.chain.from_iterable(window_pos))
	                #for 3rd person plural pronouns like 'they', if there is no re-ocurrence of pronoun 
	                #or if there are no plural nouns present in previous 2 sentences
	                #or the current sentence (upto current word) then its a mistake
	                set1 = set(criteria_word).intersection(set([t[0].lower() for t in combined_window]))
	                set2 = set(criteria_pos).intersection(set([t[1] for t in combined_window]))
	                n_antecedents = len(set1) + len(set2) 
	                if not bool(set1) and not bool(set2):
	                    mistake += 1
	                      
	            if tag[0] in male_pronouns:
	                window_pos = sent_pos[i][max(0,j-2):j]
	                window_pos.append([(t1,t2) for m,(t1,t2) in enumerate(sent_pos[i][j]) if m < k])
	                combined_window = list(itertools.chain.from_iterable(window_pos))
	                #for 3rd person male pronouns like 'he/him', if there is no re-ocurrence of male pronoun 
	                #or if there are no male names present in previous 2 sentences
	                #or the current sentence (upto current word) then its a mistake
	                if not set(male_criteria).intersection(set([t[0].lower() for t in combined_window])):
	                    mistake += 1
	            
	            if tag[0] in female_pronouns:
	                window_pos = sent_pos[i][max(0,j-2):j]
	                window_pos.append([(t1,t2) for m,(t1,t2) in enumerate(sent_pos[i][j]) if m < k])
	                combined_window = list(itertools.chain.from_iterable(window_pos))
	                #for 3rd person female pronouns like 'he/him', if there is no re-ocurrence of female pronoun 
	                #or if there are no female names present in previous 2 sentences
	                #or the current sentence (upto current word) then its a mistake
	                if not set(female_criteria).intersection(set([t[0].lower() for t in combined_window])):
	                    mistake += 1
	                    
	    d1_score.append(mistake)
	#returns the raw scores and not the mapping from 1-5
	return d1_score