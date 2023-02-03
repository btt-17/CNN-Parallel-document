import re
import argparse 
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter


def main(args):
    # コーパスファイルから文書を取得
    '''
        the format of corpus file :
        url_1 [tab] url_2 [tab] score
        score = 0: 対訳文書ではない
        score = 1: 対訳文書である
    '''
    max_len_words = args.max_len_words
    max_len_chars = args.max_len_chars
    embedding_dim = args.embedding_dim

    file = open(args.file_path)
    csvreader = csv.reader(file)
    data = []
    for row in csvreader:
        data.append(row)

    # print(data)
    # データを取得
    corpus = []
    for d in data:
        # print(d)
        if len(d) >= 2:
            url_1 = d[0].replace(" ","")
            url_2 = d[1].replace(" ","")

            corpus.append(url_1) # url_1
            corpus.append(url_2) # url_2

    ## 各レベルで埋め込みを実装する
    word_vocab = create_word_vocabs(corpus, max_len_words)
    char_vocab = create_char_vocabs(word_vocab, max_len_chars)

    word_vocab_embed = embed_vocab(word_vocab, embedding_dim)
    char_vocab_embed = embed_vocab(char_vocab, embedding_dim)


    # vocabsを保存する
    if args.vc_dir != "":
        torch.save(word_vocab_embed, f"{args.vc_dir}/word_embed_vocab.pth")
        torch.save(char_vocab_embed, f"{args.vc_dir}/char_embed_vocab.pth")

    return word_vocab_embed, char_vocab_embed




# 特別な文字で文をトークン化する
def tokenize_word(text):
    delimiter = '/jp/','/ja/','/en/','_EN','_JP', '.', '/'
    regexPattern = '|'.join(map(re.escape, delimiter))
    special_character_as_word = []
    tokens = re.split(regexPattern, text)
    words = []
    for token in tokens:
        if token != "":
            words.append(token)

    # for sp in delimiter:
    #     if sp in delimiter:
    #         special_character_as_word.append(sp)
    return special_character_as_word + words

def tokenize_char(text):
    text = text.replace('/ja', '')
    text = text.replace('/jp', '')
    text = text.replace('/en', '')
    text = text.replace('_EN', '')
    text = text.replace('_JP', '')
    text = text.replace('.', '')
    text = text.replace('/', '')
    text = text.replace('\n', '')
    return list(text)

# 語彙のvocabを作成する
def create_word_vocabs(corpus, max_len_words):
    vocab = dict()
    vocab['<UNK>'] = 0
    vocab['<PAD>'] = 1

    cnt = Counter()

    #データセットに各単語の出現頻度を求める
    for url in corpus:
        url_tok = tokenize_word(url)
        words_num = min(len(url_tok), max_len_words)
        url_tok = url_tok[:words_num]

        for word in url_tok:
            cnt['<' + word + '>'] += 1

    min_occurence = 2
    idx = 2
    for word in cnt:
        if cnt[word] >= min_occurence:
            vocab[word] = idx 
            idx += 1
    
    return vocab 

## Character Vocabulary 作成
def create_char_vocabs(corpus, max_len_subwords):
    vocab = dict()
    vocab['<UNK>'] = 0
    vocab['<PAD>'] = 1

    cnt = Counter()

    #データセットに各単語の出現頻度を求める
    for word in corpus:
        chars_num = min(len(word), max_len_subwords)
        word = word[:chars_num]
        for char in word:
            cnt[char] += 1

    min_occurence = 1
    idx = 2
    for char in cnt:
        if cnt[char] >= min_occurence:
            vocab[char] = idx 
            idx += 1
    
    return vocab 

## Embedding vocabulary
def embed_vocab(vocab, dimension):
    embeds = nn.Embedding(len(vocab), dimension)
    vocab_embed = dict()

    for token, idx in vocab.items():
        embed_tensor = torch.tensor([vocab[token]], dtype=torch.long)
        vocab_embed[token] = embeds(embed_tensor)

    return vocab_embed

## 文をwordで埋め込み
def sentence_embed_word(sentence, word_vocab, max_len_words):
    preprocesed_sent = tokenize_word(sentence)
    count = 0
    vect = None

    if len(preprocesed_sent) > max_len_words:
        preprocesed_sent = preprocesed_sent[:max_len_words]

    for word in preprocesed_sent:
        if word in word_vocab:
            word = "<" + word + ">"
            if vect is None:
                vect = word_vocab[word]
            else:
                vect = torch.cat((vect, word_vocab[word]), 0)
        else:
            if vect is None:
                vect = word_vocab['<UNK>']
            else:
                vect = torch.cat((vect, word_vocab['<UNK>']), 0)
        count += 1
        if count == max_len_words:
            break
    
    while count < max_len_words:
        vect = torch.cat((vect, word_vocab['<PAD>']), 0)
        count += 1

    return vect


def word_embed_char(word, char_vocab, max_len_subwords):
    if  word == "<UNK>" or word == "<PAD>":
        chars = []     
    else:
        chars = list(word)

    if len(chars) < max_len_subwords:
        while len(chars) < max_len_subwords:
            chars.append("<PAD>")
    else:
        chars = chars[:max_len_subwords]
    
    count = 0
    vect = None

    # print("===")
    for char in chars:
        if char in char_vocab:
            if vect is None:
                vect = char_vocab[char]
            else:
                vect = torch.cat((vect, char_vocab[char]), 0)
        else:
            if vect is None:
                vect = char_vocab['<UNK>']
            else:
                vect = torch.cat((vect, char_vocab['<UNK>']), 0)
        count += 1
        if count == max_len_subwords:
            break

    while count < max_len_subwords:
        vect = torch.cat((vect, char_vocab['<PAD>']), 0)
        count += 1

    # vect: L3 x k
    # sum pooling
    vect = torch.sum(vect, 0)
    return vect

## CW
def sentence_cw_embed_word(sentence, word_vocab, char_vocab, max_len_words, max_len_subwords):
    preprocesed_sent = tokenize_word(sentence)
    res = []
    count = 0

    if len(preprocesed_sent) > max_len_words:
        preprocesed_sent = preprocesed_sent[:max_len_words]


    for word in preprocesed_sent:
        # word = "<" + word + ">"
        if word in word_vocab:
            res.append(word_embed_char(word, char_vocab, max_len_subwords))
        else:
            res.append(word_embed_char("<UNK>", char_vocab, max_len_subwords))

        count += 1
        if count == max_len_words:
            break
    while count < max_len_words:
        res.append(word_embed_char("<PAD>", char_vocab, max_len_subwords))
        count += 1
    # L2  x k
    return torch.stack(res)

def sentence_embed_word_char(sentence, word_vocab, char_vocab, max_len_words, max_len_subwords):
    # print("word embedding")
    url_w = sentence_embed_word(sentence, word_vocab, max_len_words)
    # print("character-level embedding")
    url_cw = sentence_cw_embed_word(sentence, word_vocab, char_vocab, max_len_words, max_len_subwords)
    return torch.add(url_w, url_cw)

## 文を文字で埋め込み
def sentence_embed_char(sentence, char_vocab, max_len_chars):
    preprocesed_sent = tokenize_char(sentence)
    count = 0
    vect = None

    if len(preprocesed_sent) > max_len_chars:
        preprocesed_sent = preprocesed_sent[:max_len_chars]

    for char in preprocesed_sent:
        if char in char_vocab:
            if vect is None:
                vect = char_vocab[char]
            else:
                vect = torch.cat((vect, char_vocab[char]), 0)
        else:
            if vect is None:
                vect = char_vocab['<UNK>']
            else:
                vect = torch.cat((vect, char_vocab['<UNK>']), 0)
        count += 1
        if count == max_len_chars:
            break
    
    while count < max_len_chars:
        vect = torch.cat((vect, char_vocab['<PAD>']), 0)
        count += 1

    return vect

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len_words",dest="max_len_words",type=int,help="",default=200)
    parser.add_argument("--max_len_chars",dest="max_len_chars",type=int,help="",default=200)
    parser.add_argument("--embedding_dim",dest="embedding_dim",type=int,help="",default=32)
    parser.add_argument("--vc_dir",dest="vc_dir",help="",default=None)
    parser.add_argument("--file_path",dest="file_path",help="",default=None)

    args = parser.parse_args()
    main(args)