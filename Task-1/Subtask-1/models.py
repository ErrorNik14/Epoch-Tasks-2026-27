import torch
from torch import nn
import numpy as np


# Defining the MLP based model.
class MLPModel(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim=None, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'mlp'
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        no_input = seq_len*emb_dim if emb_dim else seq_len
        if emb_dim:
            self.embed = nn.Embedding(vocab_size, emb_dim).to(device)
            self.flat = nn.Flatten()
        self.mlp_seq = nn.Sequential(
                                nn.Linear(no_input, 128),
                                nn.Dropout(0.2),       
                                nn.ReLU(),
                                nn.Linear(128, 128),   
                                nn.Dropout(0.2),        
                                nn.ReLU(),
                                nn.Linear(128, seq_len*seq_len),  
                                #nn.Softmax()
                                nn.Tanh()
                            ).to(device)
    
    def forward(self, nums):
        if self.emb_dim:
            x = self.embed(nums)
            x = self.flat(x)
            x = self.mlp_seq(x)
            return x
        else:
            return self.mlp_seq(nums)


# Defining the RNN based model.
class RNNModel(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim=None, num_layers=1, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'rnn'
        self.emb_dim = emb_dim
        dim_input = emb_dim if emb_dim else 1
        self.seq_len = seq_len
        if emb_dim:
            self.embed = nn.Embedding(vocab_size, emb_dim).to(device) #EmbedAndConcat(no_num, 16)
        self.rnn_seq = nn.RNN(input_size=dim_input, hidden_size=128, num_layers=num_layers, batch_first=True).to(device)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(128, seq_len).to(device)
    
    def forward(self, nums): 
        if self.emb_dim:
            x = self.embed(nums)
            x = self.rnn_seq(x)[0]
            x = self.drop(x)
            x =  self.fc(x)
            return x
        else:
            x = nums.unsqueeze(-1).float()
            x = self.rnn_seq(x)[0]
            x = self.drop(x)
            x =  self.fc(x)
            return x


# Defining the LSTM based model.
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim=None, num_layers=1, bidirectional=True, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'lstm'
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        dim_input = emb_dim if emb_dim else 1
        if emb_dim:
            self.embed = nn.Embedding(vocab_size, emb_dim).to(device) #EmbedAndConcat(no_num, 16)
        self.lstm_seq = nn.LSTM(input_size=dim_input, hidden_size=128, num_layers=num_layers, bidirectional=bidirectional, batch_first=True).to(device)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(128*2 if bidirectional else 128, seq_len).to(device)
    
    def forward(self, nums): 
        if self.emb_dim:
            x = self.embed(nums)
            # print(torch.isnan(x).any(), torch.isinf(x).any())
            # raise
            x = self.lstm_seq(x)[0]
            x = self.drop(x)
            x =  self.fc(x)
            return x
        else:
            x = nums.unsqueeze(-1).float()
            x = self.lstm_seq(x)[0]
            x = self.drop(x)
            x =  self.fc(x)
            return x


# Attention layers
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, dmodel, device,*args, **kwargs):
        super().__init__(*args, **kwargs)
        posi = torch.arange(seq_len).unsqueeze(1).to(device)
        dim = torch.arange(dmodel).unsqueeze(0).to(device)
        self.register_buffer('pos_enc', torch.sin(posi/1000**(2*(dim-dim%2)/dmodel))*(1-dim%2) + torch.cos(posi/1000**(2*(dim-dim%2)/dmodel))*(dim%2) )
                    # |___ LLM used for .register_buffer() itself, not the contents
    
    def forward(self, input):
        return input + self.pos_enc


class SingleHeadedSelfAttention(nn.Module):
    def __init__(self, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.Wq = nn.Parameter(torch.randn((dmodel, dk)))
        self.Wk = nn.Parameter(torch.randn((dmodel, dk)))
        self.Wv = nn.Parameter(torch.randn((dmodel, dv)))
        self.Wf = nn.Parameter(torch.randn((dv, dmodel)))

    def forward(self, X):
        Q = torch.matmul(X,self.Wq)
        K = torch.matmul(X,self.Wk)
        V = torch.matmul(X,self.Wv)
        score = torch.softmax(torch.matmul(Q,K.transpose(-2, -1))/ np.sqrt(self.dk), dim=-1)
        attention = torch.matmul(score,V)
        out = torch.matmul(attention,self.Wf)
        return out
    
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h
        self.softmax = nn.Softmax(dim=2)
        if int(dmodel//h * h)!=dmodel:
            raise ValueError("Dimension of raw embeddings must be divisible by number of heads!")

        # LLM was used to find Xavier initialisation to combat exploding gradients from Gaussian initialisation.
        # I decided to expand on that and do an ablation on the different popular initialisation methods, observing the differences (if any).

        # self.Wq = nn.Parameter(torch.randn_like(torch.empty(h, dmodel//h, dk)))
        # self.Wk = nn.Parameter(torch.randn_like(torch.empty(h, dmodel//h, dk)))
        # self.Wv = nn.Parameter(torch.randn_like(torch.empty(h, dmodel//h, dv)))
        # self.Wf = nn.Parameter(torch.randn_like(torch.empty(dv*h, dmodel)))
        
        self.Wq = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dk))) # <--- Default baseline
        self.Wk = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dk)))
        self.Wv = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dv)))
        self.Wf = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dv*h, dmodel)))

        # self.Wq = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(h, dmodel//h, dk)))
        # self.Wk = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(h, dmodel//h, dk)))
        # self.Wv = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(h, dmodel//h, dv)))
        # self.Wf = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dv*h, dmodel)))

        # self.Wq = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(h, dmodel//h, dk)))
        # self.Wk = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(h, dmodel//h, dk)))
        # self.Wv = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(h, dmodel//h, dv)))
        # self.Wf = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(dv*h, dmodel)))

        # self.Wq = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(h, dmodel//h, dk)))
        # self.Wk = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(h, dmodel//h, dk)))
        # self.Wv = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(h, dmodel//h, dv)))
        # self.Wf = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(dv*h, dmodel)))

        # self.Wq = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h, dmodel//h, dk)))
        # self.Wk = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h, dmodel//h, dk)))
        # self.Wv = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h, dmodel//h, dv)))
        # self.Wf = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(dv*h, dmodel)))
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        # print(X.shape, X.device,'\n\n')
        X = torch.stack(torch.split(X, self.dmodel//self.h, dim=2), dim=1)
        # print(X.shape)
        Q = torch.matmul(X,self.Wq)
        K = torch.matmul(X,self.Wk)
        V = torch.matmul(X,self.Wv)
        batchscore = torch.softmax(torch.matmul(Q,K.transpose(-2,-1))/np.sqrt(self.dk), dim=-1)
        batchscore = self.dropout(batchscore)
        headattention = torch.matmul(batchscore,V)
        # headattention.shape # batchsize, heads, tokens, dmodel//heads
        attention = torch.flatten(torch.permute(headattention, (0,2,1,3)), start_dim=-2, end_dim=-1)
        # print(attention.shape)
        out = torch.matmul(attention,self.Wf)
        
        # print("headattention:",headattention[0,:,0],headattention.shape, '\n\n\n', "attention:", attention[0][0],attention.shape, '\n\n\noutput:',out[0][0],out.shape)
        # raise
        return out, batchscore



class FFNN(nn.Module):
    def __init__(self, dmodel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        no_input = dmodel
        self.mlp_seq = nn.Sequential(
                                nn.Linear(no_input, 64),       
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(64, no_input),
                                nn.Dropout(0.1),
                                nn.ReLU(),   
                            )
    
    def forward(self, input):
        out = self.mlp_seq(input)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiheadatt = MultiHeadedSelfAttention(h, dmodel, dk, dv)
        self.ffnn = FFNN(dmodel)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
    
    def forward(self, input):
        x1,att = self.multiheadatt(input)
        # print(x1.shape)
        # print(input.shape)
        # print(x1.device, input.device)
        x1 = self.norm1(x1+input)
        x2 = self.ffnn(x1)
        out = self.norm2(x2+x1)
        return out,att


# Defining the Transformer-Encoder based model.
class BERTModel(nn.Module):
    def __init__(self, N, h, dmodel, dk, dv, vocab_size, seq_len, device=None, viz=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'bert'
        self.seq_len = seq_len
        self.viz = viz
        if viz:
            self.scores = []
        no_inputs=seq_len*dmodel
        self.embed = nn.Embedding(vocab_size, dmodel).to(device)
        self.norm = nn.LayerNorm(dmodel).to(device)
        self.pos_enc = PositionalEncoding(seq_len, dmodel, device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(h, dmodel, dk, dv).to(device) for _ in range(N)])
        self.flatten = nn.Flatten()
        self.mlp_seq = nn.Sequential(
                                nn.Linear(no_inputs, 128),  
                                nn.Dropout(0.05),     
                                nn.ReLU(),
                                nn.Linear(128, 128),          
                                nn.Dropout(0.05), 
                                nn.ReLU(),
                                nn.Linear(128, seq_len*seq_len),  
                                nn.Dropout(0.05),
                                nn.Tanh()
                            ).to(device)
    
    def forward(self, input):
        if self.viz:
            self.scores = []
        x = self.embed(input)
        x = self.norm(x)
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x,att=layer(x)
            if self.viz:
                self.scores.append(att)
        x = self.flatten(x) 
        x = self.mlp_seq(x)
        return x if not self.viz else (x, self.scores) 