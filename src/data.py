from utils import *

class Hatexplain_Dataset():
    def __init__(self, data, model_path = 'bert-base-uncased', num_labels = 2, batch_size = 16, train = False):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        if num_labels == 3:
            self.label_dict = {0: 0,
                                1: 1,
                                2: 2}
        elif num_labels == 2:
            self.label_dict = {0: 1,
                                1: 0,
                                2: 1}
                                    
        self.count_dic = {}
        self.inputs, self.labels, self.attn = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels)
    def tokenize(self, sentences, padding = True, max_len = 128):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
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
    
    def process_masks(self, masks):
        mask = []
        for idx in range(len(masks[0])):
            votes = 0
            for at_mask in masks:
                if at_mask[idx] == 1: votes+=1
            if votes > len(masks)/2: mask.append(1)
            else: mask.append(0)
        return mask

    def process_data(self, data):
        sentences, labels, attn = [], [], []
        print(len(data))
        for row in data:
            word_tokens_all, word_mask_all = returnMask(row, tokenizer)
            at_mask = self.process_masks(word_mask_all)
            label = max(set(row['annotators']['label']), key = row['annotators']['label'].count)
            sentence = ' '.join(row['post_tokens'])
            sentences.append(sentence)
            labels.append(label)
            row['final_label'] = dict_label[label]
            at_mask = at_mask + [0]*(128-len(at_mask))
            attn.append(at_mask)
        inputs = self.tokenize(sentences)
        return inputs, torch.Tensor(labels), torch.Tensor(attn)
    
    def get_dataloader(self, inputs, attn, labels, train = True):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'], attn, labels)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size, drop_last=True)