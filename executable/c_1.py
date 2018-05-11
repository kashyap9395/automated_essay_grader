# -*- coding: utf-8 -*-
#subject-verb agreement
from nltk.tree import Tree

def count_c1(essay_parsed):
	person_score = []
	third_person_pronoun = ['she','he','it'] # 3rd person subject-pronouns
	subjective_pronoun = ['i','you','he','she','it', 'we','they']
	remaining_person_pronoun = set(subjective_pronoun) - set(third_person_pronoun)
	remaining_person_pronoun.remove('they')    


	number_score = []
	singular_nouns = ['NN', 'NNP']
	plural_nouns = ['NNS', 'NNPS']
	singular_verb = ['VBZ']
	plural_pronouns = ['we','they'] # subjective plural pronouns
	singular_pronouns = ['I','he','she','it']
	#Note: 'You' is ambiguous between singular and plural


	#tree = Tree.fromstring(nlp.parse(essay_tokenized[2][2]))
	for i,essay in enumerate(essay_parsed):
	    pscore = 0
	    nscore = 0
	    for j,tree_string in enumerate(essay):
	        tree = Tree.fromstring(tree_string)
	        
	        for subtree in tree.subtrees():
	            if subtree.label() == 'S':
	                NPs = [sb for sb in subtree if sb.label() == 'NP']
	                VPs = [sb for sb in subtree if sb.label() == 'VP']
	                
	                if NPs != [] and VPs != []:
	                    head_det = []
	                    head_modal = []
	                    head_noun = []
	                    head_pronoun = []
	                    head_verb = []
	           
	                    nouns = [sb.label() for sb in NPs[0] if "NN" in sb.label()]
	                    pronouns = [sb.leaves() for sb in NPs[0] if sb.label() == 'PRP']
	                    dets = [sb.leaves() for sb in NPs[0] if sb.label() == 'DT']
	                    if nouns != []:
	                        head_noun = nouns[0]
	                    if pronouns != []:
	                        head_pronoun = pronouns[0][0]
	                    if dets != []:
	                        head_det = dets[0][0]

	 
	                    verbs = [sb.label() for sb in VPs[0] if "VB" in sb.label()]
	                    if verbs != []:
	                        head_verb = verbs[0]
	                    modal_verbs = [sb.label() for sb in VPs[0] if sb.label() == "MD"]
	                    if modal_verbs != []:
	                        head_modal = modal_verbs[0]


	                    # Subject-Verb agreement  for person code 
	                    if head_noun != []:
	                        if head_noun == "NNP" and (head_verb == "VBP" or head_verb == "VBN"):
	                            pscore += 1

	                        if head_noun in plural_nouns and head_verb in singular_verb:
	                            nscore += 1

	                    if head_pronoun != []:
	                        if head_pronoun.lower() in third_person_pronoun and (head_verb == "VBP" or head_verb == "VBN"):
	                            pscore += 1

	                        if head_pronoun.lower() in remaining_person_pronoun and (head_verb == "VBZ" or head_verb == "VBN"):
	                            pscore += 1

	                        if head_pronoun.lower() in plural_pronouns and head_verb in singular_verb:
	                            nscore += 1 
	                 
	    person_score.append(pscore)
	    number_score.append(nscore)

	sv_agree = [sum(x) for x in zip(person_score, number_score)]

	    # returns penalty scores and not in range [1,5] 
	return sv_agree