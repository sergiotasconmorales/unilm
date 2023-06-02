# Script to convert introspect data from LXMERT format to VQA format

import json
import os
from os.path import join as jp

path_data_lxmert = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup/'
path_data_vqa = '/home/sergio814/Documents/PhD/code/data/VQA2/qa/'
path_output = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup_vqa_format/'
# create path_output if it doesn't exist
if not os.path.exists(path_output):
    os.makedirs(path_output)

to_process = 'val'

dict_ref = {'val': 'v2_OpenEnded_mscoco_test2015_questions'}

# Load data
with open(jp(path_data_lxmert, to_process + '.json')) as f:
    data = json.load(f)

# read vqa data to use as reference data
with open(jp(path_data_vqa, dict_ref[to_process] + '.json')) as f:
    data_ref = json.load(f)

# Convert to VQA format. I need to create a list of dictionaries with the fields: image_id, question and question_id
data_vqa = []
for entry in data:
    new_entry = {'image_id': int(entry['img_id'][-6:]), 'question': entry['sent'], 'question_id': entry['question_id']}
    data_vqa.append(new_entry)

# Replace data_ref['questions'] with data_vqa
data_ref['questions'] = data_vqa

# Save data
with open(jp(path_output, dict_ref[to_process] + '.json'), 'w') as f:
    json.dump(data_ref, f)

