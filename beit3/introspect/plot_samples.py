import os
import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jp
from PIL import Image
from tqdm import tqdm
from collections import Counter

path_imgs = '/home/sergio814/Documents/PhD/code/data/coco/images/'
path_preds = '/home/sergio814/Documents/PhD/code/beit-3/pred/introspect_zero_shot/'
inconsistent_ids_file = 'inconsistent_ids.txt'
predictions_file = 'submit_vqav2_test.json'
path_qa = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup'
path_output = jp(path_preds, 'inconsistent_samples')

# Load val data
with open(jp(path_qa, 'val.json')) as f:
    data_val = json.load(f)
    id2entry = {entry['question_id']: entry for entry in data_val}

# load predictions
with open(jp(path_preds, predictions_file)) as f:
    preds = json.load(f)
    id2ans_pred = {entry['question_id']: entry['answer'] for entry in preds}

# load txt file with inconsistent ids as a list
with open(jp(path_preds, inconsistent_ids_file)) as f:
    inconsistent_ids = f.readlines()
    inconsistent_ids = [int(x.strip()) for x in inconsistent_ids]

# iterate over inconsistent ids and plot both questions, both images, and the predicted and gt answers, as well as the relationship between the two propositions
counts_rels = {'-->': 0, '<--': 0, '<->': 0}
for id in tqdm(inconsistent_ids):
    sub_id = id
    sub_image = id2entry[sub_id]['img_id']
    sub_question = id2entry[sub_id]['sent']
    sub_answer = id2entry[sub_id]['label']
    rel = id2entry[sub_id]['rel']
    if rel in counts_rels:
        counts_rels[rel] += 1
    """
    sub_pred = id2ans_pred[sub_id]
    sub_image_path = jp(path_imgs, 'val', sub_image + '.jpg')
    sub_image = Image.open(sub_image_path)
    sub_image = np.array(sub_image)

    # get the other id
    main_id = id2entry[sub_id]['parent']
    main_image = id2entry[main_id]['img_id']
    main_question = id2entry[main_id]['sent']
    main_answer = id2entry[main_id]['label']
    main_pred = id2ans_pred[main_id]
    main_image_path = jp(path_imgs, 'val', main_image + '.jpg')
    main_image = Image.open(main_image_path)
    main_image = np.array(main_image)

    # create subplot 1 row 3 cols, plot main image and main info on the left, in the middle put only the relation, and on the right plot the sub image and sub info
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(main_image)
    axs[0].set_title(main_question + '\nGT: ' + str(main_answer) + '\nPred: ' + main_pred)
    axs[0].set_axis_off()
    axs[1].text(0.5, 0.5, rel, horizontalalignment='center', verticalalignment='center', fontsize=20)
    axs[1].set_axis_off()
    axs[2].imshow(sub_image)
    axs[2].set_title(sub_question + '\nGT: ' + str(sub_answer) + '\nPred: ' + sub_pred)
    axs[2].set_axis_off()
    # set background color to white
    fig.patch.set_facecolor('white')
    # save figure
    plt.savefig(jp(path_output, str(id) + '.png'))
    plt.close()
    """

print(counts_rels)



