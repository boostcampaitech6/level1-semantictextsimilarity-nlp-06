import yaml

from settings import * 

def load_yaml(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config 

def elapsed_time(start, end):
    elapsed = end - start 
    elapsed_min = elapsed // 60 
    elapsed_sec = elapsed - elapsed_min * 60 
    return elapsed_min, elapsed_sec 

def torch2npy(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
    npy = tensor.detach().cpu().numpy()
    return npy

def log(args, epoch, elapsed_min, elapsed_sec, train_loss, valid_loss):
    print(f'Epoch: [{epoch+1}/{args.num_epochs}]\tElapsed Time: {elapsed_min}m {elapsed_sec:.2f}s')
    print(f'Train Loss: {train_loss:.4f}\tValid Loss: {valid_loss:.4f}')
