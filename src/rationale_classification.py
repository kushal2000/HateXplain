import numpy as np
from datasets import list_datasets, load_dataset
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import random
import re

from models import Rationale_With_Labels
from data import Hatexplain_Dataset

RANDOM_SEED = 42
MODEL_PATH = 'bert-base-uncased'
RATIONALE_IMP_FACTOR = 2.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = load_dataset('hatexplain', split = ['train', 'validation', 'test'])
train_dataset = dataset[0], valid_dataset = dataset[1], test_dataset = dataset[2]

train_data_source = Hatexplain_Dataset(train_dataset, num_labels = 2, model_path = MODEL_PATH, train = True)
val_data_source = Hatexplain_Dataset(valid_dataset, num_labels = 2, model_path = MODEL_PATH)
test_data_source = Hatexplain_Dataset(test_dataset, num_labels = 2, model_path = MODEL_PATH)

def get_predicted(preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat

def evaluate_classifier(test_dataloader, model):
    model.eval()
    y_preds, y_test = np.array([]), np.array([])

    total = 0
    correct = 0
    pred = []
    label = []
    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_attn, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long(), batch[3].to(device).long()
        with torch.no_grad():        
            ypred, _, _ = model(b_input_ids, b_input_mask, b_attn)
        ypred = ypred.cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        try:
            y_preds = np.hstack((y_preds, get_predicted(ypred)))
            y_test = np.hstack((y_test, label_ids))
        except:
            y_preds, y_test = ypred, label_ids
    
    print(classification_report(y_test, y_preds))
    f1 = f1_score(y_test, y_preds, average = 'macro')

    return f1, y_test, y_preds

def evaluate_rationales(test_dataloader, model):
    model.eval()
    y_preds, y_test, y_mask = None, None, None

    total = 0
    correct = 0
    pred = []
    label = []
    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_attn, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long(), batch[3].to(device).long()
        with torch.no_grad():        
            _, _, logits = model(b_input_ids, b_input_mask, b_attn)
        ypred = logits.cpu().numpy()
        label_ids = b_attn.to('cpu').numpy()
        mask = b_input_mask.to('cpu').numpy()
        try:
            y_preds = np.hstack((y_preds, ypred))
            y_test = np.hstack((y_test, label_ids))
            y_mask = np.hstack((y_mask, mask))
        except:
            y_preds, y_test, y_mask = ypred, label_ids, mask

    for i in range(y_mask.shape[0]):
        for j in range(len(y_mask[i])):
            # if y_mask[i][j] == 0: break
            if np.argmax(y_preds[i][j]) == y_test[i][j]: correct += 1
            pred.append(np.argmax(y_preds[i][j]))
            label.append(y_test[i][j])
            total += 1

    acc = correct/total

    print(classification_report(label, pred))
    return (acc, correct, total, pred, label)
 
def train(training_dataloader, validation_dataloader, model, filepath = None, weights = None, learning_rate = 2e-5, epochs = 4, print_every = 10):
    total_steps = len(training_dataloader) * epochs
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    best_weighted_f1 = 0
    best_model = None
    # current_epoch, best_weighted_f1 = load_metrics(filepath, model, optimizer)

    criterion = nn.CrossEntropyLoss()

    for epoch_i in tqdm(range(0, epochs)):
        model.train()
        for step, batch in enumerate(training_dataloader):
            b_input_ids, b_input_mask, b_attn, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long(), batch[3].to(device).long()
            
            ypred, loss, logits = model(b_input_ids, b_input_mask, b_attn)

            loss = loss + criterion(ypred, b_labels)

            # print(outputs.logits)
            if step%print_every == 0:
                print(loss.item())
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
 
        print('### Validation Set Stats')
        weighted_f1, ytest, ypred = evaluate_classifier(validation_dataloader, model)
        (acc, correct, total, pred, label) = evaluate_rationales(validation_dataloader, model)
        print("  Macro F1: {0:.3f}".format(weighted_f1))
        if weighted_f1 > best_weighted_f1:
            best_weighted_f1 = weighted_f1
            best_model = copy.deepcopy(model)
            # save_metrics(filepath, epoch_i, model, optimizer, weighted_f1)
        
    return best_model

model = Rationale_With_Labels(768, 2).to(device)
best_model = train(train_data_source.DataLoader, val_data_source.DataLoader, model, epochs = 5)