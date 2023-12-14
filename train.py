import time 
import argparse 

import torch 
from tqdm.auto import tqdm 

import wandb 

import torch 
from torch import nn, optim 
from torch.optim.lr_scheduler import OneCycleLR

from utils import *
from settings import * 
from data_utils import * 
from models import Model, get_trainloader, get_validloader

def evaluate(args, model, valid_loader, criterion):
    valid_loss = 0
    with torch.no_grad():
        model.eval()
        for batch in valid_loader:
            X = {'input_ids': batch[0]['input_ids'].to(args.device), 
                 'type_ids': batch[0]['type_ids'].to(args.device), 
                 'mask': batch[0]['mask'].to(args.device)}
            y = batch[1].to(args.device)
            
            pred_y = model(**X).squeeze()
            loss = criterion(pred_y, target=y)
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)
    return valid_loss

def predict(args, model, test_loader):
    pred_ys = []
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            X = {'input_ids': batch['input_ids'].to(args.device), 
                 'type_ids': batch['type_ids'].to(args.device), 
                 'mask': batch['mask'].to(args.device)}
            
            pred_y = model(**X).squeeze()
            pred_ys.append(pred_y)
    return list(round(float(i)*5, 1) for i in torch.cat(pred_ys))


def trainer(args, model, train_loader, valid_loader, criterion, optimizer, scheduler):
    if args.wandb:
        wandb.init(
        project="Level01", 
        name=f'{args.model_name.split("/")[-1]}-{args.batch_size}-{args.lr}.pt',
        config={
            'learning_rate': args.lr, 
            'batch_size': args.batch_size,
            'model_name': args.model_name, 
            
        })
    train_losses, valid_losses = [], []
    best_loss = float('inf')
    count = 0
    for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
        train_loss = 0 
        model.train()
        start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            X = {'input_ids': batch[0]['input_ids'].to(args.device), 
                 'type_ids': batch[0]['type_ids'].to(args.device), 
                 'mask': batch[0]['mask'].to(args.device)}
            y = batch[1].to(args.device)

            pred_y = model(**X).squeeze()
            loss = criterion(pred_y, target=y)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # scheduler.step()
            
        train_loss /= len(train_loader)
        valid_loss = evaluate(args, model, valid_loader, criterion)
        end = time.time()
        elapsed_min, elapsed_sec = elapsed_time(start, end)
        
        log(args, epoch, elapsed_min, elapsed_sec, train_loss, valid_loss)
        if args.wandb:
            wandb.log({
                'train loss': train_loss, 
                'valid loss': valid_loss
            })
        
        if best_loss > valid_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(PARAM_DIR, f'{args.model_name.split("/")[-1]}-{args.batch_size}-{args.lr}.pt'))
            count = 0
        else:
            count += 1
            if count == args.patient:
                print(f'Early Stopping! [Patient={args.patient}]')
                break 
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    return train_losses, valid_losses 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--num_epochs', default=20)
    parser.add_argument('--max_length', default=160)
    parser.add_argument('--patient', default=20)
    parser.add_argument('--model_name', default='jhgan/ko-sroberta-sts')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    train_loader = get_trainloader(args)
    valid_loader = get_validloader(args)

        
    
    model = Model().to(args.device)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)

    scheduler = OneCycleLR(optimizer=optimizer, max_lr=0.1, steps_per_epoch=10, epochs=args.num_epochs, anneal_strategy='cos')
    criterion = nn.L1Loss()
    
    train_losses, valid_losses = trainer(args, model, train_loader, valid_loader, criterion, optimizer, scheduler)
