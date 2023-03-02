import re
import csv
#import pandas as pd
from torch.utils.data import Dataset
from constants import *


class SequenceDataset(Dataset):
    def __init__(self, answer_file_path, tokenizer, DEVICE):
        # Read JSON file and assign to headlines variable (list of strings)
        self.data_dict = []
        self.lable_set = set()
        self.device = DEVICE
        # Train Set
        file_data = []
        for file in answer_file_path:
            print(file)
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
            ref_list = q_rubric_dict[q_id]
            data_list.append(q_text)
            for t in ref_list[0:]:
                # add data
                data_list.append(t)
            data_list.append(ans_text)
            # random p_correct
            p_random_list = p_correct_random[q_id]
            for t in p_random_list[0:]:
                data_list.append(t)
            data = []
            self.lable_set.add(cat)
            data.append(cat)
            data.append(data_list)
            self.data_dict.append(data)
        print("Reading data complete")
        self.tokenizer = tokenizer
        self.tag2id = {'0': 0, '1': 1, '2': 2}
        print(self.tag2id)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        label, line_list = self.data_dict[index]
        label = self.tag2id[label]
        # print("++++++++++++++++")
        # print(label)
        input_ids, segment_ids, input_masks, labels = [], [], [], []
        for id, line in enumerate(line_list):
            tokens = self.tokenizer.tokenize(line)
            if len(tokens) > (hyperparameters['MAX_SEQ_LENGTH']-2):
                tokens = tokens[0:hyperparameters['MAX_SEQ_LENGTH']-3]
            # Add [CLS] at the beginning and [SEP] at the end of the tokens list for classification problems
            tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
            #print(tokens)
            # Convert tokens to respective IDs from the vocabulary
            input_id = self.tokenizer.convert_tokens_to_ids(tokens)

            # Segment ID for a single sequence in case of classification is 0.
            segment_id = [0] * len(input_id)

            # Input mask where each valid token has mask = 1 and padding has mask = 0
            input_mask = [1] * len(input_id)

            # padding_length is calculated to reach max_seq_length
            padding_length = hyperparameters['MAX_SEQ_LENGTH'] - len(input_id)
            input_id = input_id + [0] * padding_length
            input_mask = input_mask + [0] * padding_length
            segment_id = segment_id + [0] * padding_length
            #print(len(input_id))
            assert len(input_id) == hyperparameters['MAX_SEQ_LENGTH']
            assert len(input_mask) == hyperparameters['MAX_SEQ_LENGTH']
            assert len(segment_id) == hyperparameters['MAX_SEQ_LENGTH']
            input_ids.append(input_id)
            segment_ids.append(segment_id)
            input_masks.append(input_mask)
            labels.append(label)
        return torch.tensor(input_ids, dtype=torch.long, device=self.device), \
               torch.tensor(segment_ids, dtype=torch.long, device=self.device), \
               torch.tensor(input_masks, device=self.device), \
               torch.tensor(labels, dtype=torch.long, device=self.device)

    def set2id(self, item_set, pad=None, unk=None):
        item2id = {}
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1

        for item in item_set:
            item2id[item] = len(item2id)

        return item2id

    def id2label(self, id):
        result = "NA"
        for key, val in self.tag2id.items():
            if val == id:
                result = key
        return result
