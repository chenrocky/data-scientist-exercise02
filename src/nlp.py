# Note, in order to run this program, you will need to download wiki-news-300d-1M.vec from the fastText website
# Supporting write-up: https://medium.com/@rchen1990/clustering-and-tracking-aviation-accidents-over-time-3406ac63028e

import json
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import re
import pickle as pk
from gensim.models import KeyedVectors
from sklearn.cluster import DBSCAN
import time
from datetime import datetime


def preprocess(list_of_string):
    """
    Function to preprocess data i.e. lowercase text, replace characters

    Parameters:
    -----------
    list_of_string: list of strings

    Returns:
    --------
    processed_text: list of preprocessed string
    """
    a = int(len(list_of_string)/4)
    b = int(a*2)
    c = int(a*3)

    processed_text = []

    for pc in range(0, len(list_of_string)):
        # ----- Lowercase all text
        pc_final = list_of_string[pc].lower()

        # ----- Replacing characters and words
        pc_final = replace_text(pc_final, rep_mapping)

        # ----- Replacing all non-alphanumberic
        pc_final = re.sub("[^0-9a-zA-Z ]", "", pc_final)

        processed_text.append(pc_final)

        if pc == a:
            print('25% done')
        elif pc == b:
            print('50% done')
        elif pc == c:
            print('75% done')
        elif pc == len(list_of_string)-1:
            print('complete!')

    return processed_text


def replace_text(text, rep_dict):
    """
    Function to replace strings in the text.

    Parameters:
    -----------
    text: string in which we want to perform entity mapping
    rep_dict: dictionary with key = value to find and value = replacement word

    Returns:
    --------
    final_text: string with entity mapped
    """
    new_rep_dict = {}
    for k in sorted(rep_dict, key=len, reverse=True):
        new_rep_dict[k] = rep_dict[k]

    rep = dict((re.escape(k), v) for k, v in new_rep_dict.items())

    pattern = re.compile("|".join(rep.keys()))
    final_text = pattern.sub(lambda m: rep[re.escape(m.group())], text).strip()
    return final_text


def doc_vector(word_embedding_vectors, document):
    """
    Function to generate document embedding.

    Parameters:
    -----------
    word_embedding_vectors: pre-trained word embedding vectors
    document: document string not tokenized

    Returns:
    --------
    doc_vec: document embeddings
    """

    # ----- Remove out-of-vocabulary words
    document = [word for word in document.split() if word in word_embedding_vectors.vocab]

    # ----- Calculate centroid of word embedding vectors in the document
    doc_vec = np.mean(word_embedding_vectors[document], axis=0)

    return doc_vec


# ----- Dictionary mapping characters or words to replacements for replace_text function created above
rep_mapping = {
    "&": "and",
    "º": " degree",
    # ----- replace with space:
    "-": " ",
    "/": " ",
    "\\": " "
}


########################################################################################################################
# ----- THIS CODE BLOCK READS IN NARRATIVE DATA JSON FILES THEN CREATES AND PICKLES NARRATIVE DATAFRAME
# ----- RUN ONCE IF YOU DO NOT HAVE THE PICKLE YET THEN COMMENT OUT AFTER CREATING THE PICKLE

# # ----- Get json file names for narrative data
# narrative_file_names = [f for f in listdir('data/') if f.startswith('NarrativeData_') and isfile(join('data/', f))]
#
# # ----- Create narrative DataFrame from multiple json files
# event_id = list()
# narrative = list()
# probable_cause = list()
# for file in narrative_file_names:
#     with open('data/' + file, mode='rt') as f:
#         data = json.load(f)
#         prep_event_id = [row['EventId'] for row in data['data']]
#         prep_narrative = [row['narrative'] for row in data['data']]
#         prep_probable_cause = [row['probable_cause'] for row in data['data']]
#         event_id.extend(prep_event_id)
#         narrative.extend(prep_narrative)
#         probable_cause.extend(prep_probable_cause)
# df_narrative = pd.DataFrame(
#     {'EventId': event_id,
#      'narrative': narrative,
#      'probable_cause': probable_cause
#     })
#
# # ----- Create pickle of narrative DataFrame
# df_narrative.to_pickle('data/df_narrative.pkl')
########################################################################################################################


# ----- Read in narrative data pickle
df_narrative = pd.read_pickle('data/df_narrative.pkl')
# ----- Check for duplicates of EventId in narrative data ----- no duplicates found
print('Duplicate Narrative EventId:', df_narrative.shape[0] != df_narrative['EventId'].nunique())
# ----- Count empty cells in narrative and probable_cause columns
print('narrative column has', (df_narrative['narrative'].values == "").sum(), 'blanks')
print('probable_cause column has', (df_narrative['probable_cause'].values == "").sum(), 'blanks')
# ----- Subset to only include rows where probable_cause and narrative is not blank
df_narrative_no_blanks = df_narrative.loc[(df_narrative['probable_cause'] != "") &
                                          (df_narrative['narrative'] != "")]


# ----- Read in aviation data download ----- delimiter == ' | ' ----- drop last column which is all blank
df_aviation = pd.read_csv('data/AviationData.txt', sep=" \| {0,1}", header=0).drop(['Unnamed: 31'], axis=1)
# ----- Rename 'Event Id' column to 'EventId'
df_aviation = df_aviation.rename(index=str, columns={"Event Id": "EventId"})
# ----- Check for duplicates of EventId in aviation data ----- duplicates found
print('Duplicate Aviation EventId:', df_aviation.shape[0] != df_aviation['EventId'].nunique())
# ----- Drop duplicate EventId in aviation data
df_aviation_no_dup = df_aviation.drop_duplicates(subset=['EventId'], keep=False)


# ----- Left join df_aviation_no_dup onto df_narrative_no_blanks
df_narrative_aviation = pd.merge(left=df_narrative_no_blanks, right=df_aviation_no_dup, on='EventId', how='left', indicator=True)

# ----- Subset to only include rows where EventId in aviation and narrative
df_narrative_aviation_final = df_narrative_aviation.loc[df_narrative_aviation['_merge'] == 'both']

# ----- Create additional event date columns
df_narrative_aviation_final.loc[:, 'Event Year Month'] = df_narrative_aviation_final['Event Date'].map(
        lambda x: x[6:10] + '-' + x[0:2])
df_narrative_aviation_final.loc[:, 'Event Year'] = df_narrative_aviation_final['Event Date'].apply(
        lambda x: datetime.strptime(x, '%m/%d/%Y').date().year)
df_narrative_aviation_final.loc[:, 'Event Date DT Format'] = df_narrative_aviation_final['Event Date'].apply(
        lambda x: datetime.strptime(x, '%m/%d/%Y').date())


# ----- Check for duplicates of probable_cause in merged data ----- duplicates found
print('Duplicate probable_cause:', df_narrative_aviation.shape[0] !=
      df_narrative_aviation_final.probable_cause.nunique())


# ----- Get duplicate probable_cause
pc = df_narrative_aviation_final.probable_cause
dup_pc = df_narrative_aviation_final[pc.isin(pc[pc.duplicated()])]
dup_pc.probable_cause.value_counts()


# ----- Subset data for recent years then create list of probable_cause sentences and corresponding event dates
recent_years_pc = df_narrative_aviation_final.loc[df_narrative_aviation_final['Event Year'].isin(
        [2015, 2014, 2013, 2012, 2011])]
list_probable_cause = list(recent_years_pc.probable_cause)
list_event_dates = list(recent_years_pc['Event Date DT Format'])
list_event_id = list(recent_years_pc['EventId'])


# ----- Explore probable_cause strings that contain symbols to determine how to handle in preprocessing
pc_symbols = []
for pc in range(0, len(list_probable_cause)):
    pc_final = list_probable_cause[pc].lower()
    pc_final = re.sub('[0-9a-zA-Z ]', '', pc_final)
    pc_symbols.append(pc_final)
pc_symbols_all = ''.join(set("".join(pc_symbols)))
# ----- Review example strings containing symbols
print("\n".join(s for s in list_probable_cause if "‘" in s))  # replace "‘", "'", "’", and "´" with ""
print("\n".join(s for s in list_probable_cause if "´" in s))  # replace "‘", "'", "’", and "´" with ""
print("\n".join(s for s in list_probable_cause if "’" in s))  # replace "‘", "'", "’", and "´" with ""
print("\n".join(s for s in list_probable_cause if "&" in s))  # replace "&" with "and"
print("\n".join(s for s in list_probable_cause if "º" in s))  # replace 'º' with " degrees"
print("\n".join(s for s in list_probable_cause if ">" in s))  # replace '>' with ""
print("\n".join(s for s in list_probable_cause if "_" in s))  # "The Safety Board's full report is ..."
print("\n\n".join(s for s in list_probable_cause if "The Safety Board's full report is available at" in s))
# consider replacing everything after this sub-string


# ----- Apply preprocessing function to the list of probable_cause text
list_pc_processed = preprocess(list_probable_cause)


# ----- Download https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip to /data folder
# ----- Load pre-trained fastText vectors
start_time = time.time()
word_vectors = KeyedVectors.load_word2vec_format('data/wiki-news-300d-1M.vec')
end_time = time.time()
print('time to load in pre-trained word vectors:', (end_time-start_time))
# time to load in pre-trained word vectors: 330.92481112480164

# ----- Testing pre-trained word vectors after being read in
# word_vectors.most_similar(positive=["nonsubmitted"])
# word_vectors.word_vec('aileron')


# ----- Get vector represenation of the entire probable_cause string
pc_doc_vecs = []
for doc in list_pc_processed:
    try:
        pc_doc_vec = doc_vector(word_vectors, doc)
    except:
        pc_doc_vec = np.empty(shape=(300,))

    pc_doc_vecs.append(pc_doc_vec)


# ----- Prepare pc_doc_vecs for DBSCAN
pc_doc_vecs_array = np.array(pc_doc_vecs)

# ----- DBSCAN see how different levels of espilon affect the number of clusters (n_classes)
n_classes = {}
start_time = time.time()
for i in (np.arange(0.001, 1, 0.002)):
    dbscan = DBSCAN(eps=i, min_samples=2, metric='cosine').fit(pc_doc_vecs_array)
    n_classes.update({i: len(pd.Series(dbscan.labels_).value_counts())})
end_time = time.time()
print('time to get n_classes from DBSCAN:', (end_time-start_time))
# time to get n_classes from DBSCAN: 619.0279178619385 --> note, this was for 5,433 documents

# ----- Write n_classes as pickle
# with open('data/n_classes_dict.pkl', 'wb') as f:
#     pk.dump(n_classes, f, protocol=pk.HIGHEST_PROTOCOL)

# ----- Read pickle for n_classes
with open('data/n_classes_dict.pkl', 'rb') as f:
    n_classes = pk.load(f)

# ----- DBSCAN cluster ----- epsilon of 0.013 produced the largest number of clusters
dbscan = DBSCAN(eps=0.013, min_samples=2, metric='cosine').fit(pc_doc_vecs_array)

# ----- Check out how many probable_cause are in each cluster
dbscan_unique_elements, dbscan_counts_elements = np.unique(dbscan.labels_, return_counts=True)
dbscan_count_by_cluster = list(zip(dbscan_unique_elements, dbscan_counts_elements))

# ----- Create DBSCAN results DataFrame
dbscan_results = pd.DataFrame({'EventId': list_event_id, 'label': dbscan.labels_})
# ----- Left join df_aviation_no_dup onto df_narrative
dbscan_recent_years_pc = pd.merge(left=dbscan_results, right=recent_years_pc.drop(['_merge'], axis=1), on='EventId', how='left', indicator=True)

# ----- Pickle results DataFrame
dbscan_recent_years_pc.to_pickle('data/dbscan_recent_years_pc.pkl')

# ----- Read in pickle results DataFrame
dbscan_recent_years_pc = pd.read_pickle('data/dbscan_recent_years_pc.pkl')


# ----- Subset clusters, read probable_cause, and analyze commonalities
# dbscan_recent_years_pc_002 = dbscan_recent_years_pc.loc[dbscan_recent_years_pc['label'] == 2]
# dbscan_recent_years_pc_003 = dbscan_recent_years_pc.loc[dbscan_recent_years_pc['label'] == 3]
# dbscan_recent_years_pc_005 = dbscan_recent_years_pc.loc[dbscan_recent_years_pc['label'] == 5]
# dbscan_recent_years_pc_006 = dbscan_recent_years_pc.loc[dbscan_recent_years_pc['label'] == 6]
# dbscan_recent_years_pc_007 = dbscan_recent_years_pc.loc[dbscan_recent_years_pc['label'] == 7]
# dbscan_recent_years_pc_087 = dbscan_recent_years_pc.loc[dbscan_recent_years_pc['label'] == 86]
