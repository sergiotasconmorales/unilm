from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/storage/workspaces/artorg_aimi/ws_00000/sergio/beit3/tokenizer/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/storage/workspaces/artorg_aimi/ws_00000/sergio/coco",
    tokenizer=tokenizer,
    annotation_data_path="/storage/workspaces/artorg_aimi/ws_00000/sergio/coco/vqa",
)
