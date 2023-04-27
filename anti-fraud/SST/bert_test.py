import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from bert import BertClassificationModel
from dataset import MyDataset
from transformers import AdamW
from tqdm import tqdm
import random

# set random seed
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
random.seed(100)
torch.backends.cudnn.deterministic = True

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# args
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batchsize', type=int, default=64,help='')
parser.add_argument('--valid_interval', type=int, default=1,help='')
parser.add_argument('--epochs', type=int, default=100,help='')
parser.add_argument('--filepath', type=str, default="../datasets/weibo_senti_100k.csv",help='')
parser.add_argument('--lr', type=float, default=1e-4,help='')

args = parser.parse_args()

def get_df(filepath="../datasets/weibo_senti_100k.csv",batch_size = 32,check_data=False):
    df = pd.read_csv(filepath, encoding="utf8")
    df.insert(2, 'sentence', "") 
    for i in range(len(df)):
        review = df.loc[i, 'review']  # 行索引，列索引
        temp = re.sub('[^\u4e00-\u9fa5]+', '', review)  # 去除非汉字
        df.loc[i, 'sentence'] = temp
    df = df.drop('review', axis=1)
    df.to_csv('weibo_senti_100k_sentence.csv') # 保存
    X_train, X_test, y_train, y_test = train_test_split(df['sentence'].values,
                                                        df['label'].values,
                                                        train_size=0.8,
                                                        random_state=100)
    # Split train to train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=100)
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    vali_dataset = MyDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    vali_loader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False)
    if check_data:
        # print first 5 samples in train val and test
        for i, batch in enumerate(train_loader):
            print(f"Train data #{i}: {batch}")
            if i == 5:
                break
        for i, batch in enumerate(vali_loader):
            print(f"Vali data #{i}: {batch}")
            if i == 5:
                break
        for i, batch in enumerate(test_loader):
            print(f"Test data #{i}: {batch}")
            if i == 5:
                break
    return train_loader,vali_loader,test_loader

def train(train_loader,vali_loader):
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch {epoch}")
        for i,(data,labels) in enumerate(pbar):
            model.train()
            out=model(data) # [batch_size,num_class]
            loss=loss_func(out.cpu(),labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%5==0:
                out=out.argmax(dim=-1)
                acc=(out.cpu()==labels).sum().item()/len(labels)
                # set tqdm prefix
                
                pbar.set_postfix(Loss=loss.item(),ACC=acc, refresh=False)        
        if epoch%args.valid_interval==0:
            # start valid
            model.eval()
            correct = 0
            total = 0
            for i,(data,labels) in enumerate(vali_loader):
                with torch.no_grad():
                    out=model(data)
                out = out.argmax(dim=1)
                correct += (out.cpu() == labels).sum().item()
                total += len(labels)
            acc_valid = correct / total
            print(f">>> Epoch {epoch} - Val Loss: {loss.item()} - Val ACC: {acc_valid}")

def test(test_loader):
    model.eval()
    correct = 0
    total = 0
    for i,(data,labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out=model(data) # [batch_size,num_class]

        out = out.argmax(dim=1)
        correct += (out.cpu() == labels).sum().item()
        total += len(labels)
    print(f">>> The accuracy of the model on the test set is: {correct / total * 100:.2f}%")

if __name__ == '__main__':
    train_loader,vali_loader,test_loader=get_df(filepath=args.filepath,batch_size=args.batchsize,check_data=True)
    model=BertClassificationModel()
    model=model.to(device)
    optimizer=AdamW(model.parameters(),lr=args.lr)
    loss_func=torch.nn.CrossEntropyLoss()
    loss_func=loss_func.to(device)
    #train(train_loader,vali_loader)
    test(test_loader)
