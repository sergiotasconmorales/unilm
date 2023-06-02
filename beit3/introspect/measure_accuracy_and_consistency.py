# Script to measure the consistency and accuracy of the introspect predicions from BEIT-3

import json
import os
from os.path import join as jp
import numpy as np
import torch
from ast import literal_eval
from metrics import get_ans, compute_consistency_rels

path_pred = '/home/sergio814/Documents/PhD/code/beit-3/pred/introspect_zero_shot/'
path_qa = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup'
qa_name = 'val.json'
map_name = 'answer2label.txt' #! Where is this for BEIT-3. It's together with the jsonl files
pred_name = 'submit_vqav2_test.json'

rels_dict = {'-->': 0, '<--': 1, '<->':2, '---':3, 'unk': 3}

# read qa
with open(jp(path_qa, qa_name), 'r') as f:
    qa = json.load(f)
qaid2label = {e['question_id']: e['label'] for e in qa}

# read preds
with open(jp(path_pred, pred_name), 'r') as f:
    pred = json.load(f)

# load map from text file, each line is a dictionary
ans2label = []
with open(jp(path_qa, map_name), 'r') as f:
    for line in f:
        ans2label.append(literal_eval(line))

ans2label = {e['answer']: e['label'] for e in ans2label}
# add UNK  
ans2label['UNK'] = len(ans2label)

predid2ans = {e['question_id']: e['answer'] for e in pred}

# add predicted answers to qa
qa_with_rel = [e for e in qa if 'rel' in e] # i.e. all sub-questions

correct_main = torch.LongTensor(len(qa_with_rel), 1)
correct_sub = torch.LongTensor(len(qa_with_rel), 1)
question_ids_main = torch.LongTensor(len(qa_with_rel),)
question_ids_sub = torch.LongTensor(len(qa_with_rel),)
rels_int = torch.LongTensor(len(qa_with_rel), 1).zero_()
rels_onehot = torch.LongTensor(len(qa_with_rel), 4)
rels_onehot.zero_()

for i in range(correct_main.shape[0]):
    sub_question = qa_with_rel[i]['sent']
    rels_int[i] = rels_dict[qa_with_rel[i]['rel']]
    main_id = qa_with_rel[i]['parent']
    question_ids_main[i] = main_id
    sub_id = qa_with_rel[i]['question_id']
    question_ids_sub[i] = sub_id

    if len(qaid2label[main_id])<1:
        main_ans_gt = 0
    else:
        main_ans_gt = get_ans(qaid2label[main_id], ans2label)

    sub_ans_gt = get_ans(qa_with_rel[i]['label'], ans2label)

    main_ans_pred = ans2label[predid2ans[main_id]]
    sub_ans_pred = ans2label[predid2ans[sub_id]]

    correct_main[i, 0] = torch.eq(torch.tensor(main_ans_pred), torch.tensor(main_ans_gt))
    correct_sub[i, 0] = torch.eq(torch.tensor(sub_ans_pred), torch.tensor(sub_ans_gt))

rels_onehot.scatter_(1, rels_int, 1)
c, inc_idx, valid = compute_consistency_rels(correct_main, correct_sub, rels_onehot, return_indiv=True)
print('Consistency: {:.2f}%'.format(c))

dict_id_cons = {question_ids_sub[i].item(): inc_idx[i].item() for i in range(inc_idx.shape[0])}
torch.save(dict_id_cons, jp(path_pred, 'id2inc.pt'))

# save valid
dict_id_valid = {question_ids_sub[i].item(): valid[i].item() for i in range(valid.shape[0])}
torch.save(dict_id_valid, jp(path_pred, 'id2valid.pt'))

# added: compute accuracy for all questions
correct = 0
for q_id, gt_ans_scores in qaid2label.items():
    ans_gt = get_ans(gt_ans_scores, ans2label)
    ans_pred = ans2label[predid2ans[q_id]]
    correct += torch.eq(torch.tensor(ans_pred), torch.tensor(ans_gt))

print('Accuracy: {:.2f}%'.format(100*correct.item()/len(qaid2label)))


