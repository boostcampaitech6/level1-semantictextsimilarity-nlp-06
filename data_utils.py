import os 
import pandas as pd 

import torch 
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from settings import * 

class SentenceDataset(Dataset):
    def __init__(self, X, y=[], max_length=160, model_name='jhgan/ko-sroberta-sts'):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_length)
        self.sent_cols = ['sentence_1', 'sentence_2']
        self.target_col = ['label']
        
        self.X = self.df2list(X.loc[:, 'sentence_1'] + ' [SEP] ' +  X.loc[:, 'sentence_2'])
        self.y = [i/5 for i in self.df2list(y) if self.df2list(y)]
    
    def __getitem__(self, idx):
        X = self.X[idx]
        X = self.tokenizer(
            X,
            add_special_tokens=True, 
            max_length=self.max_length, 
            padding='max_length',
            truncation=True
        )
        if self.y:
            return (
                {'input_ids': torch.tensor(X['input_ids'], dtype=torch.long), 
                 'type_ids': torch.tensor(X['token_type_ids'], dtype=torch.long), 
                 'mask': torch.tensor(X['attention_mask'], dtype=torch.long)}, 
                torch.tensor(self.y[idx], dtype=torch.float)
            )
        else:
            return {'input_ids': torch.tensor(X['input_ids'], dtype=torch.long), 
                    'type_ids': torch.tensor(X['token_type_ids'], dtype=torch.long), 
                    'mask': torch.tensor(X['attention_mask'], dtype=torch.long)}
    
    def __len__(self):
        return len(self.X)
    
    def df2list(self, y):
        try:
            return y.values.tolist()
        except:
            return []
        

def get_trainloader(args):
    path = os.path.join(DATA_DIR, 'train.csv')
    dataframe = pd.read_csv(path)
    X, y = dataframe, dataframe.loc[:, 'label']
    d_set = SentenceDataset(X, y, max_length=args.max_length, model_name=args.model_name)
    return DataLoader(d_set, batch_size=args.batch_size, shuffle=True)

def get_validloader(args):
    path = os.path.join(DATA_DIR, 'dev.csv')
    dataframe = pd.read_csv(path)
    X, y = dataframe, dataframe.loc[:, 'label']
    d_set = SentenceDataset(X, y,max_length=args.max_length, model_name=args.model_name)
    return DataLoader(d_set, batch_size=args.batch_size, shuffle=False)

def get_testloader(args):
    path = os.path.join(DATA_DIR, 'test.csv')
    X = pd.read_csv(path)
    d_set = SentenceDataset(X, max_length=args.max_length, model_name=args.model_name)
    return DataLoader(d_set, batch_size=args.batch_size, shuffle=False)
