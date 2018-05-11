# -*- coding: utf-8 -*-
from Sentence import *
from Spelling import get_raw_count
from POSTagging import *
import string
import re
from  D2 import *

class Essay(object):
    """
    The main essay class that holds attibutes defined in the init function
        :param object: 
    """
    grammar = ""
    # put stanford path here
    
    def __init__(self, essay, prompt, grade,stop_words):
        self.essay = essay 
        self.prompt = prompt
        self.grade = grade
        self.stop_words = stop_words
        self.tagged = None
        self.words = []
        self.sent_list = []
        
        
    def set_words(self):
        """
        sets the words array to be used later to get spelling mistakes
            :param self: 
        """   
        self.get_sentences()
        for sent in self.sent_list:
            sent = "".join(s for s in sent if s not in ('!','.',':',',','?',';'))
            sent = sent.lower()
            self.words +=nltk.word_tokenize(sent)

    
    def get_sentences(self):
        """
        calls the sentence_list function after removing common punctuations
            :param self: 
        """
        punct = re.sub('['+string.punctuation+']', '', self.essay)
        self.sent_list =  sentence_list(punct)

    def get_length(self):
        
        return 0,0
    
    def get_spellingmistakes(self):
        """
        Gets the raw count of spelling mistakes in the given array 
            :param self: 
        """   
        raw_count = get_raw_count(self.words,self.stop_words)
        return raw_count 

    def get_tagged(self):
        """
        Tags the text with POS tags , used to generate grammar
            :param self: 
        """   
        self.tagged = get_parsed(self.sent_list)
        tagged= {}

        for sent in self.tagged:
            for tup in sent:
                dollar_removed = tup[1].replace('$','')
                if not dollar_removed in tagged and isinstance(dollar_removed, str):
                    tagged[dollar_removed] = []
                tagged[dollar_removed].append(tup[0])
            
        
        return tagged
        
        
    def get_sv_agreement(self):

        # To be filled in by kashyap

        return 0

    def get_verb_usage(self):
        """
        Returns the raw count of verb misuse, tense inconsistency and missing verb
            :param self: 
        """
        return verb_tense(self.sent_list,self.grammar)

    def get_sentence_formation(self):

        return 0

    def get_coherence(self):

        return 0

    def get_topic_relevance(self):
        return get_d2_subscores(self.essay,self.prompt,self.stop_words)
    