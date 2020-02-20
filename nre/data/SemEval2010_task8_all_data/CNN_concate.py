import torch
import torch.nn as nn
import constant as constant
import torch.nn.functional as F
"""
5_51 0.6999
glove_dim=100
lstm_hidden_dim=100
vocab_size=19197
label_size=19
max_length=90
batch_size=50
pos_embedding_dim=5
dropout=0.5
filter_list=[3]
filter_num=200
cnn_hidden_dim=100

cnn_window=3

"""
class ATTCNN(nn.Module):
    def __init__(self,pretrain_embedding_weight,hidden_dim1=200,hidden_dim2=100):
        super(ATTCNN, self).__init__()
        self.label_size=constant.label_size
        self.embedding_dim=constant.glove_dim
        self.vocab_size=constant.vocab_size
        self.max_length=constant.max_length
        self.batch_size=constant.batch_size
        self.dropout=constant.dropout
        self.pos_embedding_dim=constant.pos_embedding_dim

        self.window=constant.cnn_window


        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=constant.cnn_hidden_dim

        #embedding
        self.embedding=nn.Embedding(self.vocab_size,self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrain_embedding_weight))
        self.pos1_embedding=nn.Embedding(self.max_length*2+3,self.pos_embedding_dim)
        self.pos2_embedding = nn.Embedding(self.max_length * 2+3, self.pos_embedding_dim)
        self.emebdding_drop=nn.Dropout(self.dropout)
        self.filter_list = constant.filter_list
        self.feature_dim = self.pos_embedding_dim * 2 + self.embedding_dim * self.window
        self.filter_num = constant.filter_num
        self.convs=nn.ModuleList(
            # nn.Linear(self.embedding_dim+self.pos_embedding_dim*2,self.hidden_dim1),
           [nn.Conv2d(in_channels=1,out_channels=self.filter_num,kernel_size=(k,self.feature_dim),padding=0) for k in self.filter_list]
        )
        self.filter_dim=self.filter_num*len(self.filter_list)##卷积核的个数
        # print(self.filter_dim)
        self.conv_dropout=nn.Dropout(self.dropout)
        self.noline=nn.Sequential(
            nn.Linear(self.filter_dim,self.hidden_dim2),
            nn.Tanh()
        )
        self.line=nn.Linear(self.filter_dim+self.hidden_dim2,self.label_size)

        # self.loss=nn.CrossEntropyLoss()

    def forward(self,inputs,e1s,e1e,e2s,e2e,p1,p2):

        # embedding
        # sentence=
        # print(inputs.shape)
        # print(torch.zeros(self.batch_size,1))
        inputs=torch.cat([torch.zeros(self.batch_size,1).long(),inputs,torch.zeros(self.batch_size,1).long()],1)
        word_embed=self.embedding(inputs)
        # word_cat=torch.cat([torch.zeros(self.batch_size,).long(),inputs,torch.zeros(self.batch_size,1).long()],1)
        # print(word_embed.shape)
        WF=torch.zeros(self.batch_size,self.max_length,self.window*self.embedding_dim)

        # i=1
        # print(word_embed[:,i].shape)
        for i in range(1,self.max_length):
            # for j in range(self.window):
            WF[:,i]=torch.cat([word_embed[:,i+j] for j in range(self.window)],-1)
        # print(WF.shape)
        # PF=torch.zeros(self.batch_size,self.max_length,2,self.pos_embedding_dim)
        # start=torch.tensor(self.batch_size,1,self.embedding_dim)
        # PF[:,:,0]=pos1_embed
        pos1_embed=self.pos1_embedding(torch.tensor(p1))
        pos2_embed = self.pos2_embedding(torch.tensor(p2))
        # PF[:, :, 0] = pos1_embed
        # PF[:, :, 1] = pos2_embed
        # print(PF)


        embed=torch.cat([WF,pos1_embed,pos2_embed],-1)
        x=embed.unsqueeze(1)

        #
        x_drop=self.emebdding_drop(x)
        # print(self.conv(embed_drop).shape)
        # print(x_drop)
        # print(x_drop.squeeze(3).shape)
        z=[(conv(x_drop)).squeeze(3) for conv in self.convs]# z[idx]: batch_size x batch_max_len x (batch_max_len-knernel_size+1)
        # print(len(z))
        # print(z[1].shape)
        m=[F.max_pool1d(i,kernel_size=i.shape[-1]).squeeze(-1) for i in z]
        print(m[0].shape)
        # print(m[0].shape)
        sentence_feature=torch.cat(m,1)
        # print(sentence_feature.shape)

        sentence_feature=self.conv_dropout(sentence_feature)
        # print(sentence_feature.shape)
        g = self.noline(sentence_feature)
        # print(g.shape)
        o = torch.cat([sentence_feature, g], -1)
        # print(o.shape)
        y=self.line(o)
        # print(y)
        # print(embed.shape)
        return y