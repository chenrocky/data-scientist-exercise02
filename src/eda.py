
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
# 76133 examples; 3 features

# ----- Read in AviationData download
df_aviation = pd.read_csv('data/AviationData.txt', sep="|", header=0)
# 83,054 examples; 32 features


# ----- Explore narrative and aviation DataFrames
print(df_aviation.shape)
print(df_narrative.shape)

colnames_df_aviation = df_aviation.columns
for col in colnames_df_aviation:
    print(col, (df_aviation[col].values == np.nan).sum())
