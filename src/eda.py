import json
import pandas as pd
from pandas.io.json import json_normalize
from os import listdir
from os.path import isfile, join


# ----- Get json file names for narrative data
narrative_file_names = [f for f in listdir('data/') if f.startswith('NarrativeData_') and isfile(join('data/', f))]

# ----- Create narrative DataFrame from multiple json files
event_id = list()
narrative = list()
probable_cause = list()
for file in narrative_file_names:
    with open('data/' + file, mode='rt') as f:
        data = json.load(f)
        prep_event_id = [row['EventId'] for row in data['data']]
        prep_narrative = [row['narrative'] for row in data['data']]
        prep_probable_cause = [row['probable_cause'] for row in data['data']]
        event_id.extend(prep_event_id)
        narrative.extend(prep_narrative)
        probable_cause.extend(prep_probable_cause)
df_narrative = pd.DataFrame(
    {'EventId': event_id,
     'narrative': narrative,
     'probable_cause': probable_cause
    })


# ----- Read in AviationData download
aviation_data = pd.read_csv('data/AviationData.txt', sep="|", header=0)
# 83,054 examples
