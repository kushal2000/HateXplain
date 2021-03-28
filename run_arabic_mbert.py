import numpy as np
import pandas as pd
import pickle
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import copy
from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle
import json
from sklearn.metrics import accuracy_score
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
import sys
plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams.update({'font.size': 8})
RANDOM_SEED = 42
model_path = 'xlm-roberta-base'
model_path = 'bert-base-uncased'
model_path = 'bert-base-multilingual-cased'
# model_path = 'monsoon-nlp/tamillion'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
random_seed(RANDOM_SEED, True)

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

class Dataset():
    def __init__(self, data, batch_size = 32):
        self.data = data
        # self.val_data = val_data
        self.batch_size = batch_size

        self.label_dict = {0: 0,
                            1: 1}
                                    
        self.count_dic = {}
        self.inputs, self.labels = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.labels)
        # self.train_dataloader = self.process_data(dataset_file, post_id_divisions_file, 'train')
        # self.val_dataloader = self.process_data(dataset_file, post_id_divisions_file, 'test')
        # self.test_dataloader = self.process_data(dataset_file, post_id_divisions_file, 'test')

    def ek_extra_preprocess(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent

    def tokenize(self, sentences, padding = True, max_len = 128):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        input_ids, attention_masks, token_type_ids = [], [], []
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return {'input_ids': input_ids, 'attention_masks': attention_masks}
    
    def process_data(self, data):
        sentences, labels = [], []
        # print(len(data))
        for label, sentence in zip(list(data['label']), list(data['text'])):
            label = self.label_dict[label]
            self.count_dic[label] = self.count_dic.get(label, 0) + 1
            sentence = self.ek_extra_preprocess(sentence)
            # print(sentence)
            sentences.append(sentence)
            labels.append(label)
        inputs = self.tokenize(sentences)
        return inputs, torch.Tensor(labels)
    
    def get_dataloader(self, inputs, labels, train = True):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'], labels)
        if train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)
    
    chars = ['a','d','e','g','h', 'j']
chars = ['a','c']
# chars = ['b']
splits = ['train', 'val', 'test']
key = 'b'
language = 'German_3'
df_train = pd.DataFrame()


chars = ['a','d','e','g','h', 'j']
chars = ['a','b']
# chars = ['b']
splits = ['train', 'val', 'test']
keys = ['a', 'b']
language = 'Arabic_1'
df_train = pd.DataFrame()
translated = ''

for split in splits:
    for key in keys:
        for char in chars:
            df = pd.read_csv(translated+'delimit_dataset/only_hate/'+str(split)+'/'+language+str(char)+'_full.csv', sep=',', engine='python')
            df = df.rename(columns={'Unnamed: 0': 'id'})
            ids = pd.read_csv('delimit_dataset/DE-LIMIT/Dataset/ID Mapping/train/'+language+key+'_full.csv', sep=',', engine='python')
            df = df[df['id'].isin(ids['id'])]
            # print(char,len(ids), df_train)
            df_train = pd.concat([df_train, df])
df_train = df_train.drop_duplicates(subset =['id'])

df_val = pd.DataFrame()
for split in splits:
    for key in keys:
        for char in chars:
            df = pd.read_csv(translated+'delimit_dataset/only_hate/'+str(split)+'/'+language+str(char)+'_full.csv', sep=',', engine='python')
            df = df.rename(columns={'Unnamed: 0': 'id'})
            ids = pd.read_csv('delimit_dataset/DE-LIMIT/Dataset/ID Mapping/val/'+language+key+'_full.csv', sep=',', engine='python')
            df = df[df['id'].isin(ids['id'])]
            # print(char,len(ids), df_train)
            df_val = pd.concat([df_val, df])
df_val = df_val.drop_duplicates(subset =['id'])

df_test = pd.DataFrame()
for split in splits:
    for key in keys:
        for char in chars:
            df = pd.read_csv(translated+'delimit_dataset/only_hate/'+str(split)+'/'+language+str(char)+'_full.csv', sep=',', engine='python')
            df = df.rename(columns={'Unnamed: 0': 'id'})
            ids = pd.read_csv('delimit_dataset/DE-LIMIT/Dataset/ID Mapping/test/'+language+key+'_full.csv', sep=',', engine='python')
            df = df[df['id'].isin(ids['id'])]
            # print(char,len(ids), df_train)
            df_test = pd.concat([df_val, df])
df_test = df_test.drop_duplicates(subset =['id'])

train_data, val_data, test_data = Dataset(df_train), Dataset(df_val), Dataset(df_test)

class SC_weighted_BERT(nn.Module):
    def __init__(self, model_path):
        super(SC_weighted_BERT, self).__init__()
        self.num_labels = 2
        self.weights=[1.0795518,  0.82139814, 1.1678787]
        self.train_att= True
        self.lam = 100
        self.num_sv_heads = 6
        self.sv_layer = 11
        self.bert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        #self.softmax=nn.Softmax(config.num_labels)
        # self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        #logits = self.softmax(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        # if labels is not None:
        # loss_funct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
        # loss_logits =  loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
        # loss= loss_logits
            # if(self.train_att):
                
            #     loss_att=0
            #     for i in range(self.num_sv_heads):
            #         attention_weights=outputs[1][self.sv_layer][:,i,0,:]
            #         loss_att +=self.lam*masked_cross_entropy(attention_weights,attention_vals,attention_mask)
            #     loss = loss + loss_att
        outputs = outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
 
def get_predicted(preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat
 
def evaluate(test_dataloader, model):
    model.eval()
    y_preds, y_test = np.array([]), np.array([])

    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long()
        with torch.no_grad():        
            ypred = model(b_input_ids, b_input_mask)
        ypred = ypred[0].cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        y_preds = np.hstack((y_preds, get_predicted(ypred)))
        y_test = np.hstack((y_test, label_ids))

    weighted_f1 = f1_score(y_test, y_preds, average='macro')
    report = classification_report(y_test, y_preds)
    print(report)
    return weighted_f1, y_preds, y_test
 
def train(training_dataloader, validation_dataloader, model, filepath = None, weights = None, learning_rate = 2e-5, epochs = 1, print_every = 10):
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
    if weights == None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    for epoch_i in tqdm(range(0, epochs)):
        model.train()
        for step, batch in enumerate(training_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long()
            
            outputs = model(b_input_ids, b_input_mask)
            loss = criterion(outputs[0], b_labels)
 
            if step%print_every == 0:
                print(loss.item())
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
 
        print('### Validation Set Stats')
        weighted_f1, ypred, ytest = evaluate(validation_dataloader, model)
        print("  Macro F1: {0:.3f}".format(weighted_f1))
        if weighted_f1 > best_weighted_f1:
            best_weighted_f1 = weighted_f1
            best_model = model
            # save_metrics(filepath, epoch_i, model, optimizer, weighted_f1)
    # del model
    del best_model
    return model

try:
    with open('mbert_arabic.pkl', 'rb') as f:
        data_dict = pickle.load(f)
except:
    data_dict = {}
    with open('mbert_arabic.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

random_seeds = [766, 1009, 994, 89, 571]
# random_seeds = [766]
attentions = [0, 0.001, 0.01, 0.1, 1, 10, 100]
# attentions = [0, 0.01,  1, 100]
N = [0, 20, 50, 100, 200, 300, 400]
n = int(sys.argv[1])
for random_seed in random_seeds:
    if n!=0:
        df = pd.concat([df_train[df_train['label']==0].sample(n=n, random_state = random_seed),
                    df_train[df_train['label']==1].sample(n=n, random_state = random_seed)],
                    ignore_index = True)
        data = Dataset(df)
    for attn in attentions:
        if attn not in data_dict: data_dict[attn] = {}
        if n not in data_dict[attn]: data_dict[attn][n] = {}
        if random_seed in data_dict[attn][n]: continue
        model = SC_weighted_BERT(model_path)
        # model.load_state_dict(torch.load('./Saved/bert-base-uncased_11_6_3_100/pytorch_model.bin', 'cpu'))
        model.to(device)
        if attn!=0:
            model.load_state_dict(torch.load('./Saved_mbert/bert-base-uncased_11_6_2_'+str(attn)+'/model.pt', 'cpu'))
        else:
            model.load_state_dict(torch.load('./Saved_mbert/bert-base-uncased_11_6_2_0.0/model.pt', 'cpu'))
        model.to(device)
        if n!=0:
            model = train(data.DataLoader, val_data.DataLoader, model, None, epochs = 4)
        f1, ypreds, ytest = evaluate(test_data.DataLoader, model)
        acc = accuracy_score(ytest, ypreds)
        with open('mbert_arabic.pkl', 'rb') as f:
            data_dict = pickle.load(f)

        if attn not in data_dict: data_dict[attn] = {}
        if n not in data_dict[attn]: data_dict[attn][n] = {}
        if random_seed in data_dict[attn][n]: continue

        data_dict[attn][n][random_seed] = {'f1': f1, 'acc':acc}
        print(attn, n, data_dict[attn][n])
        with open('mbert_arabic.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
        del model