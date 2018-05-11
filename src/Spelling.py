# -*- coding: utf-8 -*-
from Sentence import *
from nltk.corpus import wordnet as wn



def get_raw_count(words,stop_words):
    """
    This function takes in the essay and the stop words and returns the total number of
    spelling mistakes 
        :param words: 
        :param stop_words:
        @return Count

        
        @Current complexity : O(sentences * number of words) 
    """
    count = 0
    for word in words:
        if not wn.synsets(word):
            if word not in stop_words:
                count+=1 
    return count

def get_raw_count_with_dict(essay,stop_words,us_dict,uk_dict):
    """
    Same like the above function but an additional dictionary check
    Not in use
        :param essay: 
        :param stop_words: 
        :param us_dict: 
        :param uk_dict: 
    """

    count = 0
    
    return count