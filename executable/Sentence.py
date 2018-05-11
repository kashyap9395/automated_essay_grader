# -*- coding: utf-8 -*-
# Sentence processor

import nltk

def sentence_list(text):
    """
    returns tokenized sentence list for the given text
        :param text: 
    """
    lis = nltk.sent_tokenize(text)[0].split("\n")
    #print(lis)
    return lis
