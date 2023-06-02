# Script to convert introspect data from LXMERT format to VQA format

import json
import os
from tqdm import tqdm
import numpy as np
from os.path import join as jp

path_data_lxmert = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup/'
path_data_vqa = '/home/sergio814/Documents/PhD/code/data/VQA2/qa/'
path_output = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup_vqa_format/'
# create path_output if it doesn't exist
if not os.path.exists(path_output):
    os.makedirs(path_output)

to_process = 'train'

dict_ref = {'train_q': 'v2_OpenEnded_mscoco_train2014_questions', 'train_a': 'v2_mscoco_train2014_annotations',}

def get_ans(dict_labels):
    # finds best answer: the one with highest score
    ans_list = list(dict_labels.keys())
    ans_scores = list(dict_labels.values())
    index_max = np.argmax(ans_scores)
    return ans_list[index_max]

# Load data
with open(jp(path_data_lxmert, to_process + '.json')) as f:
    data = json.load(f)

# read vqa questions to use as reference data
with open(jp(path_data_vqa, dict_ref[to_process+'_q'] + '.json')) as f:
    data_ref_q = json.load(f)

# read vqa annotations to use as reference data
with open(jp(path_data_vqa, dict_ref[to_process+'_a'] + '.json')) as f:
    data_ref_a = json.load(f)
    id2ans = {e['question_id']: e['answers'] for e in data_ref_a['annotations']}

# Convert to VQA format. I need to create a list of dictionaries with the fields: image_id, question and question_id
data_vqa_q = []
data_vqa_a = []
for entry in tqdm(data):
    # create entry for question
    new_entry_q = { 'image_id': int(entry['img_id'][-6:]), 
                    'question': entry['sent'], 
                    'question_id': entry['question_id']
                    }
    data_vqa_q.append(new_entry_q)
    # create entry for annnotation
    if entry['question_id'] in id2ans:
        answers = id2ans[entry['question_id']]
    else:
        answers = [{'answer': get_ans(entry['label']), 'answer_confidence': 'yes', 'answer_id': i} for i in range(1,11)]
    new_entry_a = { 'image_id': int(entry['img_id'][-6:]), 
                    'question_id': entry['question_id'], 
                    'question_type': entry['question_type'],
                    'answer_type': entry['answer_type'], 
                    'answers': answers,
                    'multiple_choice_answer': get_ans(entry['label'])
                    }
    data_vqa_a.append(new_entry_a)

# Replace data_ref_q['questions'] with data_vqa_q
data_ref_q['questions'] = data_vqa_q
data_ref_a['annotations'] = data_vqa_a

# Save data
with open(jp(path_output, dict_ref[to_process+'_q'] + '.json'), 'w') as f:
    json.dump(data_ref_q, f)

with open(jp(path_output, dict_ref[to_process+'_a'] + '.json'), 'w') as f:
    json.dump(data_ref_a, f)

