# -*- coding: utf-8 -*-
import re
from nltk import pos_tag as ptag
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
import nltk



def get_parsed(sent_list):
    """
    Returns the POS tagged sentences
    http://www.nltk.org/howto/parse.html
    http://www.nltk.org/book/ch08.html
    http://www.nltk.org/_modules/nltk/parse/sr.html
        :param sent_list: 
    """
    parsed_sent = []
    for sent in sent_list:
        sent = re.sub('[\,.!?()``:-;'']', '', sent)
        parsed = ptag(nltk.word_tokenize(sent))
        parsed_sent.append(parsed)
        
    return parsed_sent


def verb_tense(sent_list,grammar):
    """
    Loads the grammar and checks the essay for verb misuse, tense inconsistency and misisng verb
    Current Parser: Bottom Up Chart
        :param sent_list: 
        :param grammar: 
    """
    count = 0
    cfg_grammar = nltk.CFG.fromstring(grammar)
    for sent in sent_list:
        li = nltk.word_tokenize(sent)
        #parser = nltk.ShiftReduceParser(cfg_grammar,trace=False)
        parser = nltk.parse.BottomUpChartParser(cfg_grammar)
        chart = parser.chart_parse(li)
        if ((len(list(chart.parses(cfg_grammar.start()))))) is 0:
            count+=1
        # if len(list(parser.parse(li)))==0:
        #     count+=1
            
    return count


def check_rules(tags):
    """
    Regex to check for Verb formations - not in use 
        :param tags: 
    """
    pattern = re.compile(r'''
        (PRP\sMD\sVB\sVBN\sVBG)|
        (PRP\sVBP\sVBN\sVBG)# Rule 5: PRP->VBP->VBN->VBG (Present perfect cont)
        (PRP\sMD\sVB\sVBG)| # Rule 11: PRP->MD->VB->VBG (Future Cont)
        (PRP\sMD\sVB\sVBN)|
        (PRP\sVBP\sVBG)| # Rule 2: PRP->VBP->VBG (Prsent continuos)
        (PRP\sVBZ\sVBG)| # Rule 3: PRP->VBZ->VBG (Present continuous)
        (PRP\sVBP\sVBN)| # Rule 4: PRP->VBP->VBN (Present perfect)
        (PRP\sVBD\sVBG)| #Rule 7 : PRP->VBD->VBG (past continuos)
        (PRP\sVBP\sVBN)| #Rule 8 : PRP->VBP->VBN (past perf)
        (PRP\sVBD\sVBN)| # Rule 9: PRP-VBD-VBN (past perfect cont)
        (PRP\sMD\sVB)| # Rule 10 : PRP->PRP->MD->VB (simple future)
        (PRP\sVBD)|#Rule 1 : start with PRP and followed by VBP (Simple Present)
        (PRP\sVBP)| # Rule 6: PRP->VBP (Simple past)
      
    ''', re.VERBOSE)
    if  pattern.search(tags) is not None:
        li = [s for s in pattern.search(tags).groups() if s is not None]
        
        if li:
            return li
        else:
            return 0
    return 0

     


