
import json
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


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


# ----- Read in NarrativeData pickle
df_narrative = pd.read_pickle('data/df_narrative.pkl')

# ----- Read in AviationData download ; delimiter == ' | ' ; drop last column which is all blank
df_aviation = pd.read_csv('data/AviationData.txt', sep=" \| {0,1}", header=0).drop(['Unnamed: 31'], axis=1)
df_aviation['Event Year Month'] = df_aviation['Event Date'].map(lambda x: x[6:10] + '-' + x[0:2])
df_aviation['State'] = df_aviation['Location'].map(lambda x: str(x)[-2:])


# ----- Get count of rows and columns
print(df_narrative.shape)
# 76133 examples ; 3 features

print(df_aviation.shape)
# 83,054 examples ; 31 original features ; 1 derived feature


# ----- Get count of null or blank
colnames_df_narrative = df_narrative.columns
for col in colnames_df_narrative:
    print(col, (df_narrative[col].values == "").sum())

colnames_df_aviation = df_aviation.columns
for col in colnames_df_aviation:
    print(col, (df_aviation[col].isnull().sum()))


df_aviation['Event Id'].nunique()
# 81,841 so there are duplicates
df_aviation['Investigation Type'].value_counts()
# Accident: 79,690; Incident: 3,361
df_aviation['Accident Number'].nunique()
# 83,054 so all unique
event_year_month = df_aviation['Event Year Month'].value_counts().sort_index()


