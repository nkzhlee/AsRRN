'''
 This is a AsRRN Python training script.
'''

# system package
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm, trange
#import ml_metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# this project
from dataModule import SequenceDataset
from constants import *
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import WarmupLinearSchedule
from BertModules import BertClassifier
import wandb

def train(args):
    DEVICE = args.device
    print(DEVICE)
    wandb.init(project="AsRRN", entity="zhaohuilee", config=config_dictionary)
    random.seed(123)
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load the data
    train_dataset = SequenceDataset(train_file, tokenizer, DEVICE)
    test_dataset = SequenceDataset(test_file, tokenizer, DEVICE)
    dataset_size = len(train_dataset)
    testset_size = len(test_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(hyperparameters['validation_split'] * dataset_size))
    shuffle_dataset = True

    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=validation_sampler)

    # synchronization
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    print('Training Set Size {}, Validation Set Size {}'.format(len(train_indices), len(val_indices)))
    print('Testing Set Size {}'.format(testset_size))

    # Load BERT default config object and make necessary changes as per requirement
    config = BertConfig(hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        num_labels=hyperparameters['NUM_LABELS'])

    # Create our custom BERTClassifier model object
    model = BertClassifier(config, DEVICE)
    model.to(DEVICE)
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # Adam Optimizer with very small learning rate given to BERT
    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': hyperparameters['lr_bert']},
        {'params': model.o.parameters(), 'lr': hyperparameters['lr']},
        {'params': model.f.parameters(), 'lr': hyperparameters['lr']},
        {'params': model.g.parameters(), 'lr': hyperparameters['lr']}
    ])
    # Learning rate scheduler
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=hyperparameters['WARMUP_STEPS'],
                                     t_total=len(train_loader) // hyperparameters['GRADIENT_ACCUMULATION_STEPS'] * hyperparameters['NUM_EPOCHS'])
    training_acc_list, validation_acc_list = [], []
    model.zero_grad()
    epoch_iterator = trange(int(hyperparameters['NUM_EPOCHS']), desc="Epoch")
    # model
    best_acc = 0
    best_ckp_path = ''
    for epoch in epoch_iterator:
        print("Training Epoch: {}".format(epoch+1))
        train_loss = 0.0
        # Training Loop
        y_true = list()
        y_pred = list()
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            model.train(True)
            # Here each element of batch list refers to one of [input_ids, segment_ids, attention_mask, labels]
            inputs = {
                'input_ids': batch[0].squeeze(0).to(DEVICE),
                'token_type_ids': batch[1].squeeze(0).to(DEVICE),
                'attention_mask': batch[2].squeeze(0).to(DEVICE)
            }
            # print("input_ids: {}".format(batch[0].squeeze(0).size()))
            # print("token_type_ids: {}".format(batch[1].squeeze(0).size()))
            # print("attention_mask: {}".format(batch[2].squeeze(0).size()))
            labels = batch[3].squeeze(0)
            labels = [labels[0]]
            labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
            # print("a_id: {}".format(a_id))
            # print("labels: {}".format(labels.size()))
            logits = model(**inputs)
            # print("h_t: {}".format(len(h_t.tolist())))
            loss = criterion(logits, labels) / hyperparameters['GRADIENT_ACCUMULATION_STEPS']
            loss.backward()
            train_loss += loss.item()
            if (step + 1) % hyperparameters['GRADIENT_ACCUMULATION_STEPS'] == 0:
                scheduler.step()
                optimizer.step()
                model.zero_grad()
            _, predicted = torch.max(logits.data, 1)
            y_pred += list(predicted.data.cpu().numpy())
            y_true += list(labels.data.cpu().numpy())
        train_acc = accuracy_score(y_true, y_pred)
        # Validation Loop
        with torch.no_grad():
            val_loss = 0
            model.train(False)
            val_y_true = list()
            val_y_pred = list()
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                inputs = {
                    'input_ids': batch[0].squeeze(0).to(DEVICE),
                    'token_type_ids': batch[1].squeeze(0).to(DEVICE),
                    'attention_mask': batch[2].squeeze(0).to(DEVICE)
                }

                labels = batch[3].squeeze(0)
                labels = [labels[0]]
                labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
                logits = model(**inputs)
                _, predicted = torch.max(logits.data, 1)
                val_y_pred += list(predicted.data.cpu().numpy())
                val_y_true += list(labels.data.cpu().numpy())
                vloss = criterion(logits, labels) / hyperparameters['GRADIENT_ACCUMULATION_STEPS']
                val_loss += vloss.item()
                # break
            val_acc = accuracy_score(val_y_true, val_y_pred)
        print('Training Accuracy {} - Validation Accurracy {}'.format(
            train_acc, val_acc))
        print('Training loss {} - Validation Loss {}'.format(
            train_loss, val_loss))
        if val_acc > best_acc:
            best_acc = val_acc
            with open(
                    'checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch)), 'wb'
            ) as f:
                torch.save(model.state_dict(), f)
            best_ckp_path = 'checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch))
        # test
        with torch.no_grad():
            model.train(False)
            y_true = list()
            y_pred = list()
            test_iterator = tqdm(test_loader, desc="Test Iteration")
            predict_labels = []
            for step, batch in enumerate(test_iterator):
                inputs = {
                    'input_ids': batch[0].squeeze(0).to(DEVICE),
                    'token_type_ids': batch[1].squeeze(0).to(DEVICE),
                    'attention_mask': batch[2].squeeze(0).to(DEVICE)
                }
                labels = batch[3].squeeze(0)
                labels = [labels[0]]
                labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
                logits = model(**inputs)
                _, predicted = torch.max(logits.data, 1)
                true_label = test_dataset.id2label(labels.tolist()[0])
                # print(true_label)
                pre_label = test_dataset.id2label(predicted.tolist()[0])
                # print(pre_label)
                # print(len(h_t.tolist()))
                # record the predict label and hidden representations
                pred_idx = torch.max(logits, 1)[1]
                y_true += list(labels.data.cpu().numpy())
                y_pred += list(pred_idx.data.cpu().numpy())
            acc = accuracy_score(y_true, y_pred)
            print("Test acc is {} ".format(acc))
            wandb.log(
                {"Train loss": train_loss, "Val loss": val_loss,
                 "Train Acc": train_acc, "Val Acc": val_acc,
                 "test Acc": acc},
            )

    with torch.no_grad():
        model.train(False)
        y_true = list()
        y_pred = list()
        print("start to test at {} ".format(best_ckp_path))
        model.load_state_dict(torch.load('./' + best_ckp_path))
        model.eval()
        test_iterator = tqdm(test_loader, desc="Test Iteration")
        predict_labels = []
        for step, batch in enumerate(test_iterator):
            inputs = {
                'input_ids': batch[0].squeeze(0).to(DEVICE),
                'token_type_ids': batch[1].squeeze(0).to(DEVICE),
                'attention_mask': batch[2].squeeze(0).to(DEVICE)
            }
            labels = batch[3].squeeze(0)
            labels = [labels[0]]
            labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
            logits = model(**inputs)
            _, predicted = torch.max(logits.data, 1)
            true_label = test_dataset.id2label(labels.tolist()[0])
            # print(true_label)
            pre_label = test_dataset.id2label(predicted.tolist()[0])
            # print(pre_label)
            # print(len(h_t.tolist()))
            # record the predict label and hidden representations
            pred_idx = torch.max(logits, 1)[1]
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
        print(len(y_true), len(y_pred))
        acc = accuracy_score(y_true, y_pred)
        #print("Quadratic Weighted Kappa is {}".format(metrics.quadratic_weighted_kappa(y_true, y_pred)))
        print("Test acc is {} ".format(acc))
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        wandb.log({"final_test Acc": acc})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='debug_cpt',
                        help='ckp_name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()

