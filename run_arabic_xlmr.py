import numpy as np
import pandas as pd
import pickle
from my_utils import *
import sys

RANDOM_SEED = 42
model_path = sys.argv[2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed(RANDOM_SEED, True)
text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    segmenter="twitter", 
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

chars = ['a','d','e','g','h', 'j']
splits = ['train', 'val', 'test']
keys = ['a','d','e','g','h', 'j']
language = sys.argv[1]

df_train = pd.DataFrame()
translated = sys.argv[5]

for split in splits:
    for key in keys:
        for char in chars:
            try:
                df = pd.read_csv('only_hate/'+str(split)+'/'+language+str(char)+'_full_tr.csv', sep=',', engine='python')
                df = df.rename(columns={'Unnamed: 0.1': 'id'})
                ids = pd.read_csv('delimit_dataset/DE-LIMIT/Dataset/ID Mapping/train/'+language+key+'_full.csv', sep=',', engine='python')
                df = df[df['id'].isin(ids['id'])]
                df_train = pd.concat([df_train, df])
            except: 
                continue
df_train = df_train.drop_duplicates(subset =['id'])

df_val = pd.DataFrame()
for split in splits:
    for key in keys:
        for char in chars:
            try:
                df = pd.read_csv('only_hate/'+str(split)+'/'+language+str(char)+'_full.csv', sep=',', engine='python')
                df = df.rename(columns={'Unnamed: 0.1': 'id'})
                ids = pd.read_csv('delimit_dataset/DE-LIMIT/Dataset/ID Mapping/val/'+language+key+'_full.csv', sep=',', engine='python')
                df = df[df['id'].isin(ids['id'])]
                df_train = pd.concat([df_train, df])
            except: 
                continue
df_val = df_val.drop_duplicates(subset =['id'])

df_test = pd.DataFrame()
for split in splits:
    for key in keys:
        for char in chars:
            try:
                df = pd.read_csv('only_hate/'+str(split)+'/'+language+str(char)+'_full.csv', sep=',', engine='python')
                df = df.rename(columns={'Unnamed: 0.1': 'id'})
                ids = pd.read_csv('delimit_dataset/DE-LIMIT/Dataset/ID Mapping/test/'+language+key+'_full.csv', sep=',', engine='python')
                df = df[df['id'].isin(ids['id'])]
                df_train = pd.concat([df_train, df])
            except: 
                continue
df_test = df_test.drop_duplicates(subset =['id'])

train_data, val_data, test_data = Dataset(df_train, translated = translated), Dataset(df_val, translated = translated), Dataset(df_test, translated = translated)

data_dict_path = 'Results/'+model_path+language+'.pkl'
try:
    with open(data_dict_path, 'rb') as f:
        data_dict = pickle.load(f)
except:
    data_dict = {}
    with open(data_dict_path, 'wb') as f:
        pickle.dump(data_dict, f)

attentions = [0, 0.001, 0.01, 0.1, 1, 10, 100]
n = int(sys.argv[3])
random_seed = int(sys.argv[4])
model_folder = sys.argv[6]

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
        model.load_state_dict(torch.load(model_folder'/bert-base-uncased_11_6_2_'+str(attn)+'/model.pt', 'cpu'))
    else:
        model.load_state_dict(torch.load(model_folder+'/bert-base-uncased_11_6_2_0.0/model.pt', 'cpu'))
    model.to(device)
    if n!=0:
        model = train(data.DataLoader, val_data.DataLoader, model, None, epochs = 4)
    f1, ypreds, ytest = evaluate(test_data.DataLoader, model)
    acc = accuracy_score(ytest, ypreds)
    with open(data_dict_path, 'rb') as f:
        data_dict = pickle.load(f)

    if attn not in data_dict: data_dict[attn] = {}
    if n not in data_dict[attn]: data_dict[attn][n] = {}
    if random_seed in data_dict[attn][n]: continue

    data_dict[attn][n][random_seed] = {'f1': f1, 'acc':acc}
    print(attn, n, data_dict[attn][n])
    with open(data_dict_path, 'wb') as f:
        pickle.dump(data_dict, f)
    del model