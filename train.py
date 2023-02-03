from URLNet import *
from preprocess import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import argparse
import csv

torch.manual_seed(1)

cuda = torch.device('cuda')     # Default CUDA device


def train_and_evaluate(model, criterion, optimizer,train_data, valid_data, params=None):
    # best_valid_loss = float('inf')
    best_valid_acc = 0
    print(f"|\tEpoch\t|\tTrain Loss\t|\tTrain Acc\t|\tValid Loss\t|\tValid Acc\t|")
    for epoch in range(params.epochs):
        train_loss, train_acc = train(model, train_data, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_data, criterion)

        # if valid_acc <  best_valid_loss:
        #     best_valid_loss = valid_loss 
        #     torch.save(model.state_dict(), f"{params.model_dir}/best_checkpoint.pt")

        if valid_acc >  best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f"{params.model_dir}/best_checkpoint.pt")
        
        print(f"|\t{epoch+1}\t|\t{train_loss:.3f}\t|\t{train_acc:.3f}\t|\t{valid_loss:.3f}\t|\t{valid_acc:.3f}|")


def train(model, train_data, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    count = 1

    for x_batch, y_batch in train_data:
        
        # Clean gradientes
        optimizer.zero_grad()

        # print(y_batch)
        y_batch =  y_batch.type(torch.FloatTensor)

        # Feed the model
        y_pred = model(x_batch[0], x_batch[1])

        if len(y_pred) > 1:
            y_pred =  y_pred.squeeze().float()
        else:
            y_pred = y_pred.float()

        y_pred.requires_grad = True
        
 
        epoch_acc += binary_accuracy(y_pred, y_batch)

        # Loss calculation
        loss = criterion(y_pred, y_batch)

        # Gradients calculation
        # print(y_pred)
        loss.backward(retain_graph=True)

        epoch_loss += loss.item()

        optimizer.step()
    
    return epoch_loss / len(train_data), epoch_acc / len(train_data)
        
        
def binary_accuracy(preds, y):
    softmax = nn.Softmax(dim=0)
    
    # round predictions to the closest integer
    rounded_preds = torch.round(softmax(preds.float()))
    correct = (rounded_preds == y).float()
    return correct.sum() / len (correct)

def evaluate(model, valid_data, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for x_batch, y_batch in valid_data:

            y_batch =  y_batch.type(torch.FloatTensor)
            y_preds = model(x_batch[0], x_batch[1])
            if len(y_preds) > 1:
                y_preds =  y_preds.squeeze().float()
            else:
                y_preds = y_preds.float()
           

            loss = criterion(y_preds, y_batch)

            epoch_acc += binary_accuracy(y_preds, y_batch)
            epoch_loss += loss.item() 

    return epoch_loss / len(valid_data), epoch_acc / len(valid_data)

class DatasetMaper(Dataset):
    def __init__(self, x, y):
      self.x = x
      self.y = y
      
    def __len__(self):
      return len(self.x)
      
    def __getitem__(self, idx):
      return self.x[idx], self.y[idx]

def load_data(file_path, word_vocab_embed, char_vocab_embed, max_len_words, max_len_chars, max_len_subwords,mode):
    data = []
    file = open(file_path)
    csvreader = csv.reader(file)
    for row in csvreader:
        data.append(row)
    
    data_x = []
    data_x_embed = []
    data_y = []
   
    # print(data)

    #データを取得
    # print("Getting data ....")
    for i in range(len(data)):
        url_1 = data[i][0].replace(" ","")
        url_2 = data[i][1].replace(" ","")
        score = int(data[i][2])
        data_x.append([url_1, url_2])
        data_y.append(score)

    # print("Embedding ...")
    # for row in data_x:
    for i in range(len(data_x)):
        # print("sentence ",i+1)
        row = data_x[i]
        url_1 = row[0]
        url_2 = row[1]

        # Character-level embedding
        url_1_embed_char = sentence_embed_char(url_1, char_vocab_embed, max_len_chars)
        url_2_embed_char = sentence_embed_char(url_2, char_vocab_embed, max_len_chars)

        # Word-level embedding:
        if mode == 1 or mode == 2 or mode == 3:
            url_1_embed_word = sentence_embed_word(url_1, word_vocab_embed, max_len_words)
            url_2_embed_word = sentence_embed_word(url_2, word_vocab_embed, max_len_words)
        
        if mode == 4 or mode == 5:
            url_1_embed_word = sentence_embed_word_char(url_1, word_vocab_embed, char_vocab_embed, max_len_words, max_len_subwords)
            url_2_embed_word = sentence_embed_word_char(url_2, word_vocab_embed, char_vocab_embed, max_len_words, max_len_subwords)
        data_x_embed.append([torch.cat((url_1_embed_char, url_2_embed_char)), torch.cat((url_1_embed_word, url_2_embed_word))])
    
    return data_x_embed, data_y

def main(args):
    ## Vocabularysをロード
    print("=> Loading vocab")
    word_vocab_embed = torch.load(f"{args.vocab_dir}/word_embed_vocab.pth")
    char_vocab_embed = torch.load(f"{args.vocab_dir}/char_embed_vocab.pth")

    # 学習データを取得
    print("=> Loading train data .....")
    train_x, train_y = load_data(args.train_file, word_vocab_embed, char_vocab_embed,
                                 args.max_len_words, args.max_len_chars, args.max_len_subwords, args.mode)
    
    # 評価データを取得
    print("=> Loading validation data .....")
    valid_x, valid_y = load_data(args.valid_file, word_vocab_embed, char_vocab_embed,
                                 args.max_len_words, args.max_len_chars, args.max_len_subwords, args.mode)

    # モデル訓練が用いるデータを準備
    train = DatasetMaper(train_x, train_y)
    valid = DatasetMaper(valid_x, valid_y)

    train_data = DataLoader(train, batch_size=args.batch_size,shuffle=True)
    valid_data = DataLoader(valid, batch_size=args.batch_size,shuffle=True)

    # OptimizerとLoss functionを定義
    model = URLNet(args, word_vocab_embed, char_vocab_embed)
    criterion = nn.BCEWithLogitsLoss()

    if args.optim == "adam":
        optimizer = optim.Adam(model.parameters(),lr=args.lr)
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # モデルを学習
    train_and_evaluate(model, criterion, optimizer, train_data, valid_data, args)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len_words",dest="max_len_words",type=int,help="",default=200)
    parser.add_argument("--max_len_chars",dest="max_len_chars",type=int,help="",default=200)
    parser.add_argument("--max_len_subwords",dest="max_len_subwords",type=int,help="",default=30)
    parser.add_argument("--embedding_dim",dest="embedding_dim",type=int,help="",default=32)
    parser.add_argument("--learning_rate","--lr",dest="lr",type=float,help="",default=0.001)
    parser.add_argument("--optim",dest="optim",help="",default="adam")
    parser.add_argument("--epochs",dest="epochs",type=int,help="",default=10)
    parser.add_argument("--batch_size",dest="batch_size",type=int,help="",default=128)
    parser.add_argument("--num_filters",dest="num_filters",type=int,help="",default=256)
    parser.add_argument("--dropout",dest="dropout",type=float,help="",default=0.2)
    parser.add_argument("--filter_sizes",dest="filter_sizes",nargs="+",type=int,help="",default=None)
    parser.add_argument("--mode",dest="mode",type=int,help="",default=1)
    parser.add_argument("--vocab_dir",dest="vocab_dir",help="", default=None)
    parser.add_argument("--train_file",dest="train_file",help="",default=None)
    parser.add_argument("--valid_file",dest="valid_file",help="",default=None)
    parser.add_argument("--model_dir",dest="model_dir",help="",default=None)


    args = parser.parse_args()
    main(args)