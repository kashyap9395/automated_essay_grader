# -*- coding: utf-8 -*-
# C - iii part
from nltk.tree import Tree

def count_c3(essay_parsed):
    sent_form = []
    for i,essay in enumerate(essay_parsed):
        mistake = 0 
        for j,tree_string in enumerate(essay):
            flag = False
            tree = Tree.fromstring(tree_string)
            
            # for checking if the first label is NP or a VP or a PP
            # fragments with no verb, sometimes begin with NP 
            # fragments with no subject sometimes begin with VP or PP
            label = [sb.label() for sb in tree.subtrees()][1]

            if label[0] == 'NP' or label[0] == 'VP' or label[0] == 'PP':
                mistake += 1
                
            visited = []
            for subtree in tree.subtrees():
                visited.append(subtree.label())
                if subtree.label() == 'SBAR' or subtree.label() == 'FRAG':
                    if 'S' not in visited:
                        mistake += 1
                        break
            
            # checks for fragments with a mising verb            
            for subtree in tree.subtrees():
                visited.append(subtree.label())
            for each in visited:
                if "VB" in each or each == "MD":
                    flag =True

            if flag == False:
                mistake += 1
        
        sent_form.append(mistake)
    # Returns the raw scores and not the mapping from 1-5    
    return sent_form 