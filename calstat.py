from transformers import AutoTokenizer, AutoModelForSequenceClassification
import csv
import torch
from torch.utils.data import DataLoader, Dataset
from constants import *

class ColSTATDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer, device):
        # Read JSON file and assign to headlines variable (list of strings)
        self.data_dict = []
        self.device = device
        self.lable_set = set()
        file_data = []
        for file in dataset_file_path:
            with open(file, encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                file_header = next(csv_reader)
                for row in csv_reader:
                    file_data.append(row)
        for row in file_data:
            data_list = []
            a_id = row[0]
            cat = row[-1]
            q_id = row[2]
            context_text = q_context_dict[q_id]
            data_list.append(context_text)
            q_text = q_text_dict[q_id]
            ans_text = row[1]
            ref_list = correct_ref_dict[q_id][0:4] + part_ref_dict[q_id][0:4] + in_ref_dict[q_id][0:4]
            data_list.append(q_text)
            for t in ref_list[0:]:
                # add data
                data_list.append(t)
            data_list.append(ans_text)
            data = []
            self.lable_set.add(cat)
            data.append(cat)
            data.append(data_list)
            self.data_dict.append(data)
        self.tokenizer = tokenizer
        #self.tag2id = self.set2id(self.lable_set)
        self.tag2id = {'0': 0, '1': 1, '2': 2}
        print(self.tag2id)
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        label, lines = self.data_dict[index]
        label = self.tag2id[label]
        ct, q = lines[0], lines[1]
        input_ids, attention_masks = [], []
        tokenized_ct = self.tokenizer(ct, padding="max_length", truncation=True, max_length=param['max_length'])
        input_ids.append(tokenized_ct["input_ids"])
        attention_masks.append(tokenized_ct["attention_mask"])
        tokenized_q = self.tokenizer(q, padding="max_length", truncation=True, max_length=param['max_length'])
        input_ids.append(tokenized_q["input_ids"])
        attention_masks.append(tokenized_q["attention_mask"])
        for line in lines[2:]:
            #tokenized_data = self.tokenizer(line, padding="max_length", truncation=True, return_tensors="pt")
            new_line = CLS_TOKEN + line + SEP_TOKEN + q + SEP_TOKEN + ct
            tokenized_data = self.tokenizer(new_line, padding="max_length", truncation=True, max_length=param['max_length'])
            input_id = tokenized_data["input_ids"]
            attention_mask = tokenized_data["attention_mask"]
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
        return {
            "input_ids":  torch.tensor(input_ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=self.device),
            "label": label,
        }

    def set2id(self, item_set, pad=None, unk=None):
        item2id = {}
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1

        for item in item_set:
            item2id[item] = len(item2id)

        return item2id
