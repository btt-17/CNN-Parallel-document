import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import preprocess

torch.manual_seed(1)

class URLNet(nn.Module):
    def __init__(self, params, word_vocab, char_vocab):
        super().__init__()

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        # self.char_word_vocab = char_word_vocab
        
        '''
            params:
                - num_filters: the nummber of filters in each filter size
                - dropoout: the value of dropout
                - filter_sizes: [3, 4, 5, 6]
                - mode: (default = 1)
                    1: only character-based CNN
                    2: only word-based CNN
                    3: character and word CNN
                    4: character-level word CNN
                    5: character and character-level word CNN
                
                - max_len_words: 
                    The maxium number of words in a URL. The URL is either truncated
                    or padded with a <PAD> token to reach this length (default = 200)
                
                - max_len_chars:
                    The maximum number of characters in a URL. The URL is either truncated
                    or padded with a <PAD> character to reach this length (default = 200)
                
                - max_len_subwords:
                    The maximum number of characters in each word in URL. Each word is either
                    truncated or padded with a <PAD> character to reach this length (default = 20)
                
                - embedding_dim:
                    Dimension size of word and character embedding (default = 32)     
        '''
        self.params = params

        self.convs_word = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1,
                out_channels = self.params.num_filters,
                kernel_size=(fs, self.params.embedding_dim)
            )
            for fs in self.params.filter_sizes
        ])

        self.convs_char = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1,
                out_channels = self.params.num_filters,
                kernel_size=(fs, self.params.embedding_dim)
            )
            for fs in self.params.filter_sizes
        ])

        self.dropout = nn.Dropout(self.params.dropout)
        self.relu = nn.ReLU()

    def forward(self, embedded_char, embedded_word):
        '''
            - embedded_char: the vector of sentence by embedding characters
            - embedded_word: the vector of sentence by embedding vectors
            if mode = 2 or mode = 3:
                embedded_word: only word embedding
            if mode = 4 or mode = 5:
                embdded_word: character-level word embedding
        '''
        embedded_word = embedded_word.unsqueeze(1)
        embedded_char = embedded_char.unsqueeze(1)

        ################## WORD CONVOLUTION LAYER ##################

        if  self.params.mode == 2 or \
            self.params.mode == 3 or \
            self.params.mode == 4 or \
            self.params.mode == 5:
            conved = [F.relu(conv(embedded_word)).squeeze(3) for conv in self.convs_word]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

            h_drop = self.dropout(torch.cat(pooled, dim=1))
        
        ################## CHAR CONVOLUTION LAYER ##################
        if  self.params.mode == 1 or \
            self.params.mode == 3 or \
            self.params.mode == 5:
            conved = [F.relu(conv(embedded_char)).squeeze(3) for conv in self.convs_char]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

            char_h_drop = self.dropout(torch.cat(pooled, dim=1))
        
        ################## CONCAT WORK AND CHAR BRANCH ##################
        num_filters_total = 256 * len(self.params.filter_sizes)
        if self.params.mode == 3 or self.params.mode == 5:
            ww = torch.empty(num_filters_total, 512)
            nn.init.xavier_uniform_(ww, gain=nn.init.calculate_gain('relu'))
            bw = torch.empty(512)
            nn.init.constant_(bw, 0.1)
            word_output = torch.matmul(h_drop, ww) + bw

            wc = torch.empty(num_filters_total, 512)
            nn.init.xavier_uniform_(wc, gain=nn.init.calculate_gain('relu'))
            bc = torch.empty(512)
            nn.init.constant_(bc, 0.1)
            char_output = torch.matmul(char_h_drop, wc) + bc

            conv_output = torch.cat([word_output, char_output], dim=1)

        ################## CONVOLUTION LAYER OUTPUT ##################
        if self.params.mode == 1:
            conv_output = char_h_drop

        if self.params.mode == 2 or self.params.mode == 4:
            conv_output = h_drop

    
        ################## RELU AND FC ##################
        w0 = torch.empty(1024, 512)
        nn.init.xavier_uniform_(w0, gain=nn.init.calculate_gain('relu'))
        b0 = torch.empty(512)
        nn.init.constant_(b0, 0.1)
        output0 = self.relu(torch.matmul(conv_output, w0)+b0)

        w1 = torch.empty(512, 256)
        nn.init.xavier_uniform_(w1, gain=nn.init.calculate_gain('relu'))
        b1 = torch.empty(256)
        nn.init.constant_(b1, 0.1)
        output1 = self.relu(torch.matmul(output0, w1)+b1)       

        w2 = torch.empty(256, 128)
        nn.init.xavier_uniform_(w2, gain=nn.init.calculate_gain('relu'))
        b2 = torch.empty(128)
        nn.init.constant_(b2, 0.1)
        output2 = self.relu(torch.matmul(output1, w2)+b2)  

        w = torch.empty(128, 2)
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
        b = torch.empty(2)
        nn.init.constant_(b, 0.1)

        scores = torch.matmul(output2, w)+b
        predicts = torch.argmax(scores, 1)
        
        return predicts