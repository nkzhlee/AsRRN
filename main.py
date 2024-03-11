import argparse
import wandb
import os
os.environ["WANDB__SERVICE_WAIT"] = "300"
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.optim import AdamW
from colstat import ColSTATDataset
from constants import *
from asrrn import AsRRN
import pandas as pd
torch.cuda.empty_cache()


def distance(answer, examplar):
    dist = torch.nn.functional.cosine_similarity(answer.unsqueeze(0), examplar.unsqueeze(0))
    # Minkowski
#     p = 1.5
#     diff = torch.abs(answer.unsqueeze(0) - examplar.unsqueeze(0))
#     p_norm_diff = diff ** p
#     sum_of_p_norms = torch.sum(p_norm_diff)
#     dist= torch.pow(sum_of_p_norms, 1 / p)
    # Pearson Correlation
    #tensor1 = answer.unsqueeze(0)
    #tensor2 = examplar.unsqueeze(0)
    #mean1 = torch.mean(tensor1)
    #mean2 = torch.mean(tensor2)
    #deviation1 = tensor1 - mean1
    #deviation2 = tensor2 - mean2
    #covariance = torch.sum(deviation1 * deviation2) / (tensor1.shape[0] - 1)
    #std_dev1 = torch.sqrt(torch.sum(deviation1 ** 2) / (tensor1.shape[0] - 1))
    #std_dev2 = torch.sqrt(torch.sum(deviation2 ** 2) / (tensor2.shape[0] - 1))
    #dist = covariance / (std_dev1 * std_dev2)
    # Euclidean Distance
    #dist=torch.dist(tensor1, tensor2)
    # Manhattan Distance
    #dist = torch.sum(torch.abs(tensor1 - tensor2))
    return dist/param['norm']

def smax3(a, b, c):
    mid1 = torch.max(a, b)
    return torch.max(mid1, c)

def smax4(a, b, c, d):
    mid1 = torch.max(a, b)
    mid2 = torch.max(c, d)
    return torch.max(mid1, mid2)

def contrastiveloss(targets, states):
    loss = 0.0
    #take average of sim(ref, ans)
    S_c1, S_c2,  = distance(states[3], states[-1]), distance(states[4], states[-1])
    S_c3, S_c4, = distance(states[5], states[-1]), distance(states[6], states[-1])
    S_p1, S_p2 = distance(states[7], states[-1]), distance(states[8], states[-1])
    S_p3, S_p4 = distance(states[9], states[-1]), distance(states[10], states[-1])
    #epsilon = 0.2  # the amount of noise to add
    #S_p3 = (S_p1 + S_p2)/2 + random.uniform(-epsilon, epsilon)
    S_in1, S_in2 = distance(states[11], states[-1]), distance(states[12], states[-1])
    S_in3, S_in4 = distance(states[13], states[-1]), distance(states[14], states[-1])
    S_t, S_d = 0.0, 0.0
    #mid = torch.max(S_p1, S_p2)
    i = smax4(S_in1, S_in2, S_in3, S_in4)
    p = smax4(S_p1, S_p2, S_p3, S_p4)
    c = torch.mean(torch.cat((S_c1, S_c2, S_c3, S_c4), dim=0))
    #c = (S_c1 + S_c2 + S_c3 + S_c4)/4.0
    S_d = torch.exp(i/param['temp']) + \
          torch.exp(p/param['temp']) + \
          torch.exp(c/param['temp'])
#     S_d_i = i + torch.exp(S_p1/param['temp']) + torch.exp(S_p2/param['temp']) +torch.exp(S_p3/param['temp']) +torch.exp(S_p4/param['temp']) + torch.exp(S_c1/param['temp'])+torch.exp(S_c1/param['temp'])+torch.exp(S_c1/param['temp'])+torch.exp(S_c1/param['temp'])
#     S_d_p = torch.exp(S_in1/param['temp']) + torch.exp(S_in2/param['temp']) +torch.exp(S_in3/param['temp']) +torch.exp(S_in4/param['temp']) + p + torch.exp(S_c1/param['temp'])+torch.exp(S_c2/param['temp'])+torch.exp(S_c3/param['temp'])+torch.exp(S_c4/param['temp'])
#     S_d_c = torch.exp(S_in1/param['temp']) + torch.exp(S_in2/param['temp']) +torch.exp(S_in3/param['temp']) +torch.exp(S_in4/param['temp']) + torch.exp(S_p1/param ['temp']) + torch.exp(S_p2/param['temp']) +torch.exp(S_p3/param['temp']) +torch.exp(S_p4/param['temp']) + c
    if targets[0] == 0:
        S_t = torch.exp(i/param['temp'])
        loss = -torch.log(S_t / S_d)
    if targets[0] == 1:
        #print("partial correct")
        S_t = torch.exp(p/param['temp'])
        loss = -torch.log(S_t/S_d)
    if targets[0] == 2:
        #print("correct")
        S_t = torch.exp(c/param['temp'])
        loss = -torch.log(S_t/S_d)
    #print('states: ', states.size())
    s = [S_c1.data.cpu().numpy()[0], S_c2.data.cpu().numpy()[0], S_c3.data.cpu().numpy()[0], S_c4.data.cpu().numpy()[0],
         S_p1.data.cpu().numpy()[0], S_p2.data.cpu().numpy()[0], S_p3.data.cpu().numpy()[0], S_p4.data.cpu().numpy()[0], 
         S_in1.data.cpu().numpy()[0], S_in2.data.cpu().numpy()[0], S_in3.data.cpu().numpy()[0], S_in4.data.cpu().numpy()[0]]
#     s = [S_c1.data.cpu().numpy(), S_c2.data.cpu().numpy(), S_c3.data.cpu().numpy(),S_c4.data.cpu().numpy(), 
#                     S_p1.data.cpu().numpy(), S_p2.data.cpu().numpy(), S_p3.data.cpu().numpy(), S_p4.data.cpu().numpy(),
#                              S_in1.data.cpu().numpy(), S_in2.data.cpu().numpy(), S_in3.data.cpu().numpy(), S_in4.data.cpu().numpy()]
    return loss, s



def train(args):
    wandb.init(project=PROJECT_NAME, entity=ENTITY, config=config_dictionary)
    random.seed(param['random_seed'])
    print(param)
    print(args.ckp_name)
    # model
    best_acc = 0
    best_ckp_path = ''
    DEVICE = args.device
    print(DEVICE)
    model_name = param['model_name']
    print(model_name)
    # Load the tokenizer config
    config = AutoConfig.from_pretrained(model_name)
    config.max_length = param['max_length']
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
    # Load Train dataset and split it into Train and Validation dataset
    train_dataset = ColSTATDataset(TRAIN_FILE_PATH, tokenizer, DEVICE)
    test_dataset = ColSTATDataset(TEST_FILE_PATH, tokenizer, DEVICE)
    test_dataset.tag2id = train_dataset.tag2id
    trainset_size = len(train_dataset)
    testset_size = len(test_dataset)
    shuffle_dataset = True
    validation_split = param['validation_split']
    indices = list(range(trainset_size))
    split = int(np.floor(validation_split * trainset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param['batch_size'],
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param['batch_size'],
                                             sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)


    training_acc_list, validation_acc_list = [], []

    model = AsRRN(DEVICE)
    model.to(DEVICE)
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    #scheduler = StepLR(optimizer, step_size=param['lr_step'], gamma=param['lr_gamma'])
    num_training_steps = len(train_loader) * param['epochs']
    warmup_steps = int(param['WARMUP_STEPS'] * num_training_steps)  # 10% of total training steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    model.train()
    # Training Loop
    for epoch in range(param['epochs']):
        print("Training Epoch: {} LR is {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train_loss, cc_loss, ct_loss, total_sim_socre = 0, 0, 0, 0
        model.train()
        epoch_loss = 0.0
        train_correct_total = 0
        y_true = list()
        y_pred = list()
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits, states = model(input_ids, attention_mask=attention_mask)
            #logits, loss = outputs.logits, outputs.loss
            classification_loss = criterion(logits, labels)
            contrastive_loss, sim_score = contrastiveloss(labels, states)
            if epoch > param['pre_step']:
                loss = (1 - param['Lambda']) * classification_loss + param['Lambda'] * contrastive_loss
            else:
                loss = classification_loss
            optimizer.zero_grad()
            loss.backward()
            loss = loss.data.cpu().numpy()
            if type(loss) is list: loss = loss[0]
            if type(classification_loss) is list: classification_loss = classification_loss[0]
            if type(contrastive_loss) is list: contrastive_loss = contrastive_loss[0]
            train_loss += loss
            cc_loss += classification_loss.item()
            ct_loss += contrastive_loss
            pred_idx = torch.max(logits, 1)[1]
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
            nn.utils.clip_grad_norm_(model.parameters(), param['max_norm'])  # Optional: Gradient clipping
            if (step + 1) % param['GRADIENT_ACCUMULATION_STEPS'] == 0:
                #scheduler.step()
                optimizer.step()
                model.zero_grad()
                #get_linear_schedule_with_warmup
                scheduler.step()
#             optimizer.step()
#             scheduler.step()
        # step LR
        #scheduler.step()
        train_acc = accuracy_score(y_true, y_pred)
        print('Epoch {} -'.format(epoch))
        #print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

        # Validation Loop
        val_loss, val_cc_loss, val_ct_loss, total_sim_socre = 0, 0, 0, 0
        with torch.no_grad():
            model.eval()
            val_y_true = list()
            val_y_pred = list()
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                # torch.Size([1])
                logits, states = model(input_ids, attention_mask=attention_mask)
                # logits:  torch.Size([1, 3])
                # logits, loss = outputs.logits, outputs.loss
                _, predicted = torch.max(logits.data, 1)
                val_y_pred += list(predicted.data.cpu().numpy())
                val_y_true += list(labels.data.cpu().numpy())
                classification_loss = criterion(logits, labels)
                contrastive_loss, sim_score = contrastiveloss(labels, states)
                # print("classification_loss: {}".format(classification_loss))
                # print("contrastive_loss: {}".format(contrastive_loss))
                if epoch > param['pre_step']:
                    vloss = (1 - param['Lambda']) * classification_loss + param['Lambda'] * contrastive_loss
                else:
                    vloss = classification_loss
                vloss = vloss.data.cpu().numpy()
                if type(vloss) is list: vloss = vloss[0]
                if type(classification_loss) is list: classification_loss = classification_loss[0]
                if type(contrastive_loss) is list: contrastive_loss = contrastive_loss[0]
                val_cc_loss += classification_loss.item()
                val_ct_loss += contrastive_loss
                val_loss += vloss
            val_acc = accuracy_score(val_y_true, val_y_pred)
            print('Training Accuracy {} - Validation Accurracy {}'.format(
                train_acc, val_acc))
            print('Training loss {} - Validation Loss {}'.format(
                train_loss, val_loss))
            if val_acc > best_acc:
                best_acc = val_acc
                with open(
                        './checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch)), 'wb'
                ) as f:
                    torch.save(model.state_dict(), f)
                best_ckp_path = './checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch))

        with torch.no_grad():
            test_correct_total = 0
            model.eval() 
            y_true = list()
            y_pred = list()
            test_iterator = tqdm(test_loader, desc="Test Iteration")
            for step, batch in enumerate(test_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits, states = model(input_ids, attention_mask=attention_mask)
                pred_idx = torch.max(logits, 1)[1]
                y_true += list(labels.data.cpu().numpy())
                y_pred += list(pred_idx.data.cpu().numpy())
                # break
            acc = accuracy_score(y_true, y_pred)
            print("Test acc is {} ".format(acc))
            wandb.log(
            {"Train loss": train_loss, "Train classification loss": cc_loss, "Train contrastive loss": ct_loss,
             "Val loss": val_loss, "Val classification loss": val_cc_loss, "Val contrastive loss": val_ct_loss,
             "Train Acc": train_acc, "Val Acc": val_acc, "test Acc": acc})
    print('Real Test: \n')
    with torch.no_grad():
        test_correct_total = 0
        print("start to test at {} ".format(best_ckp_path))
        model.load_state_dict(torch.load('./' + best_ckp_path))
        model.eval()
        y_true = list()
        y_pred = list()
        sim_list = list()
        test_iterator = tqdm(test_loader, desc="Test Iteration")
        for step, batch in enumerate(test_iterator):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits, states = model(input_ids, attention_mask=attention_mask)
            pred_idx = torch.max(logits, 1)[1]
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
            contrastive_loss, sim_score = contrastiveloss(labels, states)
            sim_list.append(sim_score)
        acc = accuracy_score(y_true, y_pred)
        print("Test acc is {} ".format(acc))
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        wandb.log({"final_test Acc": acc})
        # output result
 
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
