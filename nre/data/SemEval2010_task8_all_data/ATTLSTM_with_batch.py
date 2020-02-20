import torch
import torch.nn as nn
# import constant as constant
# import numpy as np
# import torch.utils.data as Data
# from tqdm import tqdm

class ATTLSTM(nn.Module):
    def __init__(self,vocab_size,label_size,embedding_size,hidden_size,pretrain_embedding_weight,batch_size,dropout=0.6):
        super(ATTLSTM, self).__init__()
        self.vocab_size=vocab_size
        self.label_size=label_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.pretrain_embedding_weight=pretrain_embedding_weight
        self.dropout=dropout
        self.batch_size=batch_size
        print()
        self.embed=nn.Embedding(vocab_size,embedding_size)
        self.embed.weight.data.copy_(torch.from_numpy(pretrain_embedding_weight))
        self.embed_drop=nn.Dropout(self.dropout)
        # self.distance_embedding = nn.Embedding(constant.max_length*2, 10)
        print(pretrain_embedding_weight.shape)
        self.lstm=nn.LSTM(input_size=self.embedding_size,hidden_size=self.hidden_size,bidirectional=True,batch_first=True,num_layers=2,dropout=self.dropout)

        self.att = SelfAttention(self.hidden_size)

        self.fc=nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size,self.label_size),
            # nn.Tanh()
        )

    def init_hidden(self):
        return (torch.zeros(4,self.batch_size, self.hidden_size),
                torch.zeros(4, self.batch_size, self.hidden_size))

    def forward(self,inputs,e1s,e1e,e2s,e2e,p1,p2):
        # inputs（B,S,W)
        # print(inputs.shape)
        # print((inputs>=19197).sum())
        word_embed=self.embed(inputs)
        # p1_distance = self.distance_embedding(torch.tensor(p1))
        # p2_distance = self.distance_embedding(torch.tensor(p2))
        embedding=word_embed
        # print(embedding.shape)
        #(B,S,E)
        # print(embedding)
        embedding_drop=self.embed_drop(embedding)
        # print(embedding_drop)
        lstm_out,(lstm_h,lstm_c)=self.lstm(embedding_drop)
        # print(lstm_out.shape)
        # lstm_out:(B,S,H)
        att_out,w=self.att(lstm_out[:,:,:self.hidden_size]+lstm_out[:,:,self.hidden_size:])

        # print(att_out.shape)
        out=self.fc(att_out)
        # print(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self,hidden_dim):
        super(SelfAttention,self).__init__()
        self.hidden_dim=hidden_dim
        self.projection=nn.Sequential(
            # nn.Tanh(),
            nn.Linear(hidden_dim,64),
            nn.ReLU(True),
            nn.Linear(64,1)

        )

    def forward(self,encoder_outputs):
        energy=self.projection(encoder_outputs)
        # print(energy.shape)
        weights=nn.functional.softmax(energy.squeeze(-1),dim=1)
        # print(weights.shape)
        # print(weights.unsqueeze(-1).shape)
        outputs=(encoder_outputs*weights.unsqueeze(-1)).sum(dim=1)
        return outputs,weights
