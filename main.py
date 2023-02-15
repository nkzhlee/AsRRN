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



def distance(answer, examplar):
    # euc_dist = torch.nn.functional.pairwise_distance(answer, examplar)
    # print(euc_dist)
    temperature = 1
    dist = torch.nn.functional.cosine_similarity(answer.unsqueeze(0), examplar.unsqueeze(0))
    # print(dist)
    # dist = torch.exp(dist/temperature)
    return dist

def contrastiveloss(targets, states):
    loss = 0.0
    #take average of sim(ref, ans)
    S_c1, S_c2 = distance(states[3], states[-1]), distance(states[4], states[-1])
    S_p1, S_p2 = distance(states[5], states[-1]), distance(states[6], states[-1])
    S_in1, S_in2 = distance(states[7], states[-1]), distance(states[8], states[-1])
    S_t, S_d = 0.0, 0.0
    #print("targets {}".format(targets[0]))
    if targets[0] == 0:
        #print("in_correct")
        loss = 0.0
    if targets[0] == 1:
        #print("partial correct")
        # print(S_c1)
        # print(S_c2)
        # print(S_p1)
        # print(S_p2)
        S_t = torch.exp((S_p1 + S_p2) / 2)
        S_d = torch.exp((S_p1 + S_p2) / 2) + torch.exp((S_c1 + S_c2) / 2)
        loss = -torch.log(S_t / S_d)
    if targets[0] == 2:
        #print("correct")
        # print(S_c1)
        # print(S_c2)
        # print(S_p1)
        # print(S_p2)
        S_t = torch.exp((S_c1 + S_c2) / 2)
        S_d = torch.exp((S_p1 + S_p2) / 2) + torch.exp((S_c1 + S_c2) / 2)
        loss = -torch.log(S_t / S_d)
    #print('states: ', states.size())
    s = [S_c1.data.cpu().numpy()[0], S_c2.data.cpu().numpy()[0],
         S_p1.data.cpu().numpy()[0], S_p2.data.cpu().numpy()[0], S_in1.data.cpu().numpy()[0], S_in2.data.cpu().numpy()[0]]
    return loss, s


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
                        hidden_dropout_prob=hyperparameters["hidden_dropout_prob"],
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
        train_loss, cc_loss, ct_loss, total_sim_socre = 0, 0, 0, 0
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
            logits, states = model(**inputs)
            # print("h_t: {}".format(len(h_t.tolist())))
            classification_loss = criterion(logits, labels) / hyperparameters['GRADIENT_ACCUMULATION_STEPS']
            # add contrastive loss
            # print("states {}".format(states.size()))
            # print("targets {}".format(targets))
            contrastive_loss, sim_score = contrastiveloss(labels, states)
            loss = (1-hyperparameters['Lambda']) * classification_loss + hyperparameters['Lambda'] * contrastive_loss
            loss.backward()
            loss = loss.data.cpu().numpy()
            if type(loss) is list: loss = loss[0]
            if type(classification_loss) is list: classification_loss = classification_loss[0]
            if type(contrastive_loss) is list: contrastive_loss = contrastive_loss[0]
            train_loss += loss
            cc_loss += classification_loss.item()
            ct_loss += contrastive_loss
            # print("classification_loss: {}".format(classification_loss))
            # print("contrastive_loss: {}".format(contrastive_loss))
            # print("loss: {}".format(loss))
            # print("sim_score: {}".format(sim_score))
            # assert 1 == 0
            if (step + 1) % hyperparameters['GRADIENT_ACCUMULATION_STEPS'] == 0:
                scheduler.step()
                optimizer.step()
                model.zero_grad()
            _, predicted = torch.max(logits.data, 1)
            y_pred += list(predicted.data.cpu().numpy())
            y_true += list(labels.data.cpu().numpy())
        train_acc = accuracy_score(y_true, y_pred)
        # Validation Loop
        val_loss, val_cc_loss, val_ct_loss, total_sim_socre = 0, 0, 0, 0
        with torch.no_grad():
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
                logits, states = model(**inputs)
                _, predicted = torch.max(logits.data, 1)
                val_y_pred += list(predicted.data.cpu().numpy())
                val_y_true += list(labels.data.cpu().numpy())
                classification_loss = criterion(logits, labels) / hyperparameters['GRADIENT_ACCUMULATION_STEPS']
                contrastive_loss, sim_score = contrastiveloss(labels, states)
                # print("classification_loss: {}".format(classification_loss))
                # print("contrastive_loss: {}".format(contrastive_loss))
                vloss = (1 - hyperparameters['Lambda']) * val_cc_loss + hyperparameters['Lambda'] * val_ct_loss
                #vloss = vloss.data.cpu().numpy()
                if type(vloss) is list: vloss = vloss[0]
                if type(classification_loss) is list: classification_loss = classification_loss[0]
                if type(contrastive_loss) is list: contrastive_loss = contrastive_loss[0]
                val_cc_loss += classification_loss.item()
                val_ct_loss += contrastive_loss
                val_loss += vloss
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
                logits, states = model(**inputs)
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
                {"Train loss": train_loss, "Train classification loss": cc_loss, "Train contrastive loss": ct_loss,
                 "Val loss": val_loss, "Val classification loss": val_cc_loss, "Val contrastive loss": val_ct_loss,
                 "Train Acc": train_acc, "Val Acc": val_acc, "test Acc": acc})
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
            logits, states = model(**inputs)
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

