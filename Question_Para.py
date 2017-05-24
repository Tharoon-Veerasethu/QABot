
# coding: utf-8

# # Query to Document and to Paragraph

# ### Importing packages

# In[1]:

# --- Importing Various packages ---
import pandas as pd
import numpy as np
import scipy.stats as scipy

# Tokenizers
from nltk.tokenize import word_tokenize, sent_tokenize
# Stopwords
from nltk.corpus import stopwords

# --- GENSIM PACKAGE ---
import gensim
from gensim.models import Word2Vec, doc2vec, Doc2Vec
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora, models, similarities
from gensim.models import KeyedVectors
from gensim.summarization.bm25 import BM25


# ### Loading Datasets

# In[2]:

data_train = pd.read_json('data/squad_train_doc.json')
data_train.rename(columns={'passages': 'documents'}, inplace=True)

# # Contains the list all titles
title_list = np.load('title_list.npy').tolist()

# Contains the dictionary of title to context
dictionary_document_context = np.load('dictionary_document_context.npy').item()

tokenized_context_and_questions = np.load('tokenized_context_and_questions.npy').tolist()

untokenized_context_and_questions = np.load('untokenized_context_and_questions.npy').tolist()


# ## Loading Models

# In[3]:

# BM25 MODEL
BM_25_model = BM25(tokenized_context_and_questions)

# TFIDF MODEL
dictionary = corpora.Dictionary.load('model_data/squad.dict')
corpus = corpora.MmCorpus('model_data/squad.mm')
TFIDF_model = gensim.models.TfidfModel.load('model_data/TFIDF_model.bin')

# Doc2Vec Model
Doc2Vec_model = gensim.models.Doc2Vec.load('model_data/Doc2Vec_model.bin')

# WMD model
WMD_model = KeyedVectors.load('model_data/WMD_model.bin')


# ## QUERY Methods

# In[4]:

def BM25(query):
    ''' Accepts a question(query) to implement BM 25 Model.
        Takes a query and word tokenizes it.
              'get_scores' - Calculates the similarity distance between the tokenized-query and the document.

        --> Returns a dataframe with Document name, Score and Rank
    '''
    scores = BM_25_model.get_scores(query.split(),1)
    BM25_dataframe = pd.DataFrame({'Document':data_train.title, 'Score_BM25':scores}).sort_values(by=['Score_BM25'],ascending=False)
    BM25_dataframe['Rank_BM25'] = [i for i in range(1, len(data_train.title)+1)]
    return BM25_dataframe


# In[5]:

def TFIDF(query):
    ''' Accepts a question(query) to implement TF-IDF Model.
        Takes a query and word tokenizes it.
        'raw_corpus_query' - The word-tokenized query is compared with the dictionary used to train the document.
            'corpus_query' - The word-id and word is converted into a corpus.The corpus is then fed to the TF-IDF model.
        'similarity_table' - Stores the TF-IDF weights which are then used to get most similiar documents.
                   'ranks' - Scipy method which compares the similarity weights and sorts is accordingly.

        --> Returns a dataframe with Document name, Score and Rank
    '''
    query_1 = []
    query_1.append(word_tokenize(query))
    raw_corpus_query = [dictionary.doc2bow(word) for word in query_1]
    corpora.MmCorpus.serialize('model_data/query3.mm',raw_corpus_query)
    corpus_query = corpora.MmCorpus('model_data/query3.mm')

    similarity_table = TFIDF_model[corpus_query]
    ranks = scipy.rankdata(similarity_table, method = 'max')
    similarity_table = list(np.array(similarity_table).flatten())
    TFIDF_dataframe = pd.DataFrame({'Document':data_train.title, 'Score_TFIDF':similarity_table}).sort_values(by=['Score_TFIDF'],ascending=False)
    TFIDF_dataframe['Rank_TFIDF'] = [i for i in range(1, len(data_train.title)+1)]
    return TFIDF_dataframe


# In[6]:

def Doc2Vec(query):
    ''' Accepts a question(query) to implement Doc2Vec Model.
        Takes a query and word tokenizes it.
           'avg_sentence' - After that the average of the sentenced words are compared with every document.
           'most_similar' - Calculates the similarity distance between the avg of tokenized-sentence with every
                            document iteratively.
        'list_doc_scores' - Returns the sorted list of comparison with each doc in ascending order.

        --> Returns a dataframe with Document name, Score and Rank(top_n, ascending order sorted)
    '''

    similarity_score_matrix , list_doc_names, list_doc_scores, list_doc_ranks, rank = [], [], [], [], 1
    avg_sentence = np.zeros((200))
    count = 0
    for word in word_tokenize(query):
        if word in Doc2Vec_model.wv.vocab:
            avg_sentence +=  Doc2Vec_model[word]
            count+=1
    if count != 0:
        avg_sentence = avg_sentence / count
    similarity_score_matrix.append(Doc2Vec_model.docvecs.most_similar([avg_sentence], topn=len(title_list)))
    for each_compared_row in similarity_score_matrix[0]:
        list_doc_names.append(each_compared_row[0])
        list_doc_scores.append(each_compared_row[1])
        list_doc_ranks.append(rank)
        rank += 1
    query_comparison_dataframe = pd.DataFrame({'Document':list_doc_names, 'Score_Doc2Vec':list_doc_scores, 'Rank_Doc2Vec':list_doc_ranks})
    return query_comparison_dataframe


# ## Query to Document Method

# In[7]:

def query_to_document(query):
    """ Takes string question and returns the name of the document which the question is likely to be present in"""

    bm25_df = BM25(query).head(n=20)         # gets the dataframe of BM25 with scores and ranks of documents
    tfidf_df = TFIDF(query).head(n=20)       # gets the dataframe of TFIDF with scores and ranks of documents
    doc2vec_df = Doc2Vec(query).head(n=20)   # gets the dataframe of Doc2Vec with scores and ranks of documents

    # combining all the dataframes
    final_df = pd.merge(pd.merge(bm25_df,tfidf_df, on=['Document'], how='outer'), doc2vec_df, on=['Document'], how='outer')
    final_df.fillna(0, inplace=True)

    # Normalising the scores between 0 and 1
    bm25_normalised = (final_df.Score_BM25 - final_df.Score_BM25.min())/(final_df.Score_BM25.max()- final_df.Score_BM25.min())
    tfidf_normalised = (final_df.Score_TFIDF-final_df.Score_TFIDF.min())/(final_df.Score_TFIDF.max()-final_df.Score_TFIDF.min())
    doc2vec_normalised = (final_df.Score_Doc2Vec-final_df.Score_Doc2Vec.min())/(final_df.Score_Doc2Vec.max()-final_df.Score_Doc2Vec.min())

    # Getting the total score based on the preious overall accuracy
    final_df['total_score'] = 0.01243557 * bm25_normalised + 0.29682442 * tfidf_normalised - 0.01673123 * doc2vec_normalised

    final_document = final_df.sort_values(by='total_score', ascending=False).Document[0]

    return final_document


# ## Document to Paragraph

# In[8]:

def document_to_paragraph(query, document):
    stop_words = set(stopwords.words("english"))
    index = 0
    sent1 = [word for word in word_tokenize(query) if word not in stop_words]
    sentences = sent_tokenize(document)
    list_distances, list_sentence_index = [], []
    for each_sentence in sentences:
        sent2 = [word for word in word_tokenize(each_sentence) if word not in stop_words]
        similarity_distance = WMD_model.wmdistance(sent1, sent2)
        list_distances.append(similarity_distance)
        list_sentence_index.append(index)
        index+=1
    WMD_Dataframe = pd.DataFrame({'Sentence': sentences, 'Sentence_Index': list_sentence_index, 'WMD_Score': list_distances}).sort_values(by=['WMD_Score'],ascending=True)
    Top8_sentences = ' '.join([sent for sent in WMD_Dataframe[0:8].Sentence])

    return Top8_sentences


# # Final Method

# In[9]:

def query_to_paragraph(query):
    document_name = query_to_document(query)
    document_context = dictionary_document_context[document_name]
    paragraph = document_to_paragraph(query=query, document=document_context)
    return paragraph


# In[10]:

query = input("Enter your query: ")


# In[11]:

paragraph = query_to_paragraph(query)


# In[12]:

print(paragraph)


# In[ ]:
