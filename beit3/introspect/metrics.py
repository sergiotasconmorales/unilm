from os.path import join as jp
import json
from venv import main
import torch
from torch.nn import ReLU
import numpy as np


def get_ans(dict_labels, ans2label):
    # finds best answer: the one with highest score
    ans_list = list(dict_labels.keys())
    ans_scores = list(dict_labels.values())
    index_max = np.argmax(ans_scores)
    if ans_list[index_max] not in ans2label:
        return ans2label['UNK']
    return ans2label[ans_list[index_max]]


def compute_consistency_rels(correct_main, correct_sub, rels, return_indiv=False):
    """Function to compute consistency taking into account the relationships between main and sub-question

    Parameters
    ----------
    correct_main : torch.Tensor
        Binary vector with as many elements as there are sub-questions with relationships. i-th entry is 1 if i-th main question was answered correcty by model.
    correct_sub : torch.Tensor
        Binary vector with as many elements as there are sub-questions with relationships.i-th entry is 1 if i-th sub question was answered correcty by model.
    rels : torch.Tensor of size [N, 4]
        One hot encoding of the relationships for each pair. Columns correspond to -->, <--, <->, ---
    """
    assert len(correct_main) == len(correct_sub)
    assert rels.shape[0] == len(correct_main)

    relu = ReLU()

    # First process <-- relationships (i.e. necessary)
    diff1 = correct_sub - correct_main
    th1 = relu(diff1.squeeze(-1))
    masked1 = th1*rels[:,1]
    necessary_term = torch.sum(masked1)

    # Now --> relations (i.e. sufficient)
    diff2 = correct_main - correct_sub
    th2 = relu(diff2.squeeze(-1))
    masked2 = th2*rels[:,0]
    sufficient_term = torch.sum(masked2)

    # finally <-> relationships (i.e. equivalent pairs)
    th3 = torch.logical_xor(correct_main, correct_sub).to(int)
    masked3 = th3.squeeze(-1)*rels[:,2]
    equivalent_term = torch.sum(masked3)

    # Consistency is defined in terms of the relationships present in the data as follows:
    total_inconsistencies = necessary_term + sufficient_term + equivalent_term
    c = 1 - total_inconsistencies/torch.sum(rels[:,:3])

    if return_indiv:
        masked = masked1 + masked2 + masked3
        return 100*c.item(), masked*rels[:,:3].sum(1), rels[:,:3].sum(1)
        #return 100*c.item(), {'<--': float(100*necessary_term/total_inconsistencies), '-->': float(100*sufficient_term/total_inconsistencies), '<->': float(100*equivalent_term/total_inconsistencies)}
    else:
        return 100*c.item()