#-*- coding: utf-8 -*-
# all imports here
import gensim  
from textblob import TextBlob
import re
from nltk.corpus import wordnet as wn
import nltk
from nltk import pos_tag as ptag
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora


def tag(prompt_string):
    action_words = []
    prompt_string = re.sub('[\,.!?()``:-;'']', '', prompt_string)
    parsed = ptag(nltk.word_tokenize(prompt_string))
    #print(parsed)
    
    for p in parsed:
        #if p[1] in ('VB','NNS','VBP','VBG','VBZ','VBD','VBN','NN','NNP','RB'):
        if p[1] not in ('TO','IN','CC','PRP$','PRP','DT'):
            action_words.append(p[0].lower())
    
    return action_words

def clean(doc,stop,exclude,lemma):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def get_d2_subscores(essay,prompt,stop):
    da = 0
    db = 0
    dc = 0
    sysnet_words = []
    action_words = tag(prompt)
    for action in action_words:
        action = wn.synsets(action)
        for synset in action:
            for lemma in synset.lemmas():
                sysnet_words.append(lemma.name())

    essay_words  = tag(essay)
    #print(essay_words)

    intersect = set(essay_words).intersection(set(sysnet_words))
    da = len(intersect)
    db = TextBlob(essay).sentiment.subjectivity
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    doc_clean = [clean(doc,stop,exclude,lemma).split() for doc in [essay]]
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)
    top = ldamodel.print_topics(num_topics=1, num_words=5)[0][1]
    dc = len(set(re.findall(r'"(.*?)"',top)).intersection(set(sysnet_words).union(set(action_words))))
    return da,db,dc





