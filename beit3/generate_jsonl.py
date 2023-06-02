from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/storage/workspaces/artorg_aimi/ws_00000/sergio/coco",
    tokenizer=tokenizer,
    annotation_data_path="/storage/workspaces/artorg_aimi/ws_00000/sergio/coco/introspect",
)
