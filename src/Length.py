# -*- coding: utf-8 -*-
def count_sent(essay_pos, essay_tokenized):
	# Reference: Nilani Aluthgedara, "Recognizing Sentence Boundaries and Boilerplate". http://honors.cs.umd.edu/reports/Nilani.pdf
	eos = []
	essay_sentences = []
	essay_sent = []
	sentence = ""
	for k,essay in enumerate(essay_pos):
	    vflag = False
	    count = 0
	    for i,tag in enumerate(essay):
	        eflag = False
	        if i == len(essay)-1:
	            count += 1
	            if tag[0][0].isalpha():
	                sentence = sentence + " " + tag[0]
	            else:
	                sentence = sentence + tag[0]
	            essay_sent.append(sentence.strip())
	            essay_sentences.append(essay_sent)
	            sentence = ""
	            essay_sent = []
	            eos.append(count)
	            break
	            
	        if "VB" in tag[1]:
	            vflag = True # verb flag
	        
	        if tag[0] in ['.','!','?',';']: # comparing with possible end-of-sentence markers
	            if i < len(essay)-1:
	                if essay[i+1][0] in ['.','!','?',';']: # handling double question-marks
	                    i += 1
	                    continue
	        
	            if vflag: # if a verb has already been encountered
	                #if essay[i+1][0][0].isupper():
	                    if essay[i+1][1] in ["NNP","NNPS"]:
	                        j = i+2
	                        v2flag = False
	                        # checking if a verb appears before next period
	                        while j < len(essay)-1:
	                            if essay[j][0] in ['.','!','?',';']:
	                                break
	                            if "VB" in essay[j][1]:
	                                v2flag = True # verb flag
	                            j += 1
	                            
	                        if v2flag:
	                            count += 1 # increment count of sentences
	                            if tag[0][0].isalpha():
	                                sentence = sentence + " " + tag[0]
	                            else:
	                                sentence = sentence + tag[0]
	                            essay_sent.append(sentence.strip())
	                            sentence = ""
	                            vflag = False
	                            eflag = True
	                    else: 
	                        count += 1
	                        if tag[0][0].isalpha():
	                            sentence = sentence + " " + tag[0]
	                        else:
	                            sentence = sentence + tag[0]
	                        essay_sent.append(sentence.strip())
	                        sentence = ""
	                        vflag = False
	                        eflag = True
	        if not eflag:
	            if tag[0][0].isalpha():
	                sentence = sentence + " " + tag[0]
	            else:
	                sentence = sentence + tag[0]

	tokenized_length = [len(x) for x in essay_tokenized]
	length = [(sum(x) / 2) for x in zip(tokenized_length, eos)]
    #Retruns the number of sentences and not SCORES
	return length