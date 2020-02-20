import torch
import torch.nn as nn
import constant as constant
import torch.nn.functional as F
# from nltk.corpus import wordnet
# from nltk import data
# data.path.append(r"D:\ruanjian\Visual Studio\Shared\Python36_64\Lib\nltk_data")
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
        self.hidden_dim2 = constant.cnn_hidden_dim

        #embedding
        self.embedding=nn.Embedding(self.vocab_size,self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrain_embedding_weight))
        self.pos1_embedding=nn.Embedding(self.max_length*2+3,self.pos_embedding_dim)
        self.pos2_embedding = nn.Embedding(self.max_length * 2+3, self.pos_embedding_dim)
        self.emebdding_drop=nn.Dropout(self.dropout)
        self.filter_list=constant.filter_list
        self.feature_dim = self.pos_embedding_dim * 2 + self.embedding_dim*self.window
        self.filter_num = constant.filter_num
        self.convs=nn.ModuleList(
            # nn.Linear(self.embedding_dim+self.pos_embedding_dim*2,self.hidden_dim1),
           [nn.Conv2d(in_channels=1,out_channels=self.filter_num,kernel_size=(k,self.feature_dim),stride=(1,self.feature_dim),padding=((k-1)//2,0)) for k in self.filter_list]
        )
        self.filter_dim=self.filter_num*len(self.filter_list)##卷积核的个数
        self.U = nn.Parameter(torch.randn(self.filter_num, self.label_size))
        self.WL = nn.Parameter(torch.randn(self.label_size, self.filter_num))
        self.conv_dropout=nn.Dropout(self.dropout)
        self.max_pool=nn.MaxPool1d(self.max_length,stride=1)
        self.noline = nn.Sequential(
            nn.Linear(self.filter_dim, self.hidden_dim2),
            nn.Tanh()
        )
        self.line=nn.Linear(self.embedding_dim*6+self.hidden_dim2,self.label_size)



    def getLextual(self,word_embed,e1s,e1e,e2s,e2e):
        # l1=torch.mean(word_embed[:,e1s:e1e+1],axis=-1)
        # print(word_embed.shape)
        # print(e1s,e1e)
        i=0
        # print(word_embed[:].shape)
        l1=torch.cat([torch.mean(word_embed[i, e1s[i]:e1e[i] + 1].unsqueeze(0),axis=1) for i in range(word_embed.shape[0])],axis=0)
        # print(l1.shape)

        l2 = torch.cat([torch.mean(word_embed[i, e2s[i]:e2e[i] + 1].unsqueeze(0),axis=1) for i in range(word_embed.shape[0])],axis=0)
        # i=0
        # print(torch.cat([word_embed[i,e1s[i]-1],word_embed[i,e1e[i]+1] ],axis=-1).shape)
        l3=torch.cat([torch.cat([word_embed[i,e1s[i]-1],word_embed[i,e1e[i]+1] ],axis=-1).unsqueeze(0) for i in range(word_embed.shape[0])],axis=0)
        # print(l3.shape)
        l4 = torch.cat([torch.cat([word_embed[i,e2s[i]-1],word_embed[i,e2e[i]+1] ],axis=-1).unsqueeze(0) for i in range(word_embed.shape[0])],axis=0)
        # print(l4.shape)
        # wordnet.
        # l5=torch.mean
        return l1,l2,l3,l4
    def input_layer(self,inputs,e1s,e1e,e2s,e2e,p1,p2):
        inputs = torch.cat([torch.zeros(self.batch_size, 1).long(), inputs, torch.zeros(self.batch_size, 1).long()], 1)
        word_embed = self.embedding(inputs)
        # word_cat=torch.cat([torch.zeros(self.batch_size,).long(),inputs,torch.zeros(self.batch_size,1).long()],1)
        # print(word_embed.shape)
        WF = torch.zeros(self.batch_size, self.max_length, self.window * self.embedding_dim)

        # i=1
        # print(word_embed[:,i].shape)
        for i in range(1, self.max_length):
            # for j in range(self.window):
            WF[:, i] = torch.cat([word_embed[:, i + j] for j in range(self.window)], -1)
        word_embed=word_embed[:,1:-1]
        # print(word_embed.shape)
        l1, l2, l3, l4 = self.getLextual(word_embed, e1s, e1e, e2s, e2e)
        pos1_embed = self.pos1_embedding(torch.tensor(p1))
        pos2_embed = self.pos2_embedding(torch.tensor(p2))

        embed = torch.cat([WF, pos1_embed, pos2_embed], -1)

        l1=l1.unsqueeze(1)
        l2 = l2.unsqueeze(1)
        # print(l1.shape)
        # print(embed.shape)
        in_e1 = F.softmax(torch.bmm(word_embed, l1.transpose(2, 1)), 1)
        # print(in_e1.shape)
        in_e2 = F.softmax(torch.bmm(word_embed, l2.transpose(2, 1)), 1)
        # print(in_e1.shape)

        in_e = (in_e1 + in_e2) / 2
        R=torch.mul(embed,in_e)
        return R
    def attention_2(self,R):
        # print(R.shape)
        RU=torch.matmul(R.transpose(2,1),self.U)
        G=torch.matmul(RU,self.WL)
        G_=F.softmax(G,dim=1)
        AP=torch.mul(R,G_.transpose(2,1))
        # print(AP.shape)
        wo=self.max_pool(AP).squeeze(-1)
        # print(self.max_pool(AP).shape)
        return wo

        # print(G.shape)
    def forward(self,inputs,e1s,e1e,e2s,e2e,p1,p2):
        R=self.input_layer(inputs, e1s, e1e, e2s, e2e, p1, p2)
        # print(R.shape)
        R=[torch.tanh(conv(R.unsqueeze(1))).squeeze(-1) for conv in self.convs[0:1]]
        # print(c.shape)
        R_=R[0]
        wo=self.attention_2(R_)
        # print(wo.shape)
        return wo,self.WL

    def predict(self,wo,WL,all_y):
        wo_norm=F.normalize(wo)
        wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, self.label_size, 1)#(batch_size,label_size,filter_num)

        emb_r=torch.mm(all_y,WL) #这里并没有什么变化(label_size,filter_num)
        delta_Sy=torch.norm(wo_norm_tile-emb_r,2,2)
        pred=torch.argmin(delta_Sy,1)
        return pred






class DistanceLoss(nn.Module):
    def __init__(self, label_size, margin=1):
        super(DistanceLoss, self).__init__()
        self.label_size = label_size
        self.margin = margin

    def forward(self, wo, WL, y, all_y):
        # in_y=torch.zeros(y.shape[0],self.label_size)
        # print(y)
        in_y = torch.zeros(y.shape[0], self.label_size).scatter_(1, y.reshape(y.shape[0],1), 1)
        # print(in_y)
        wo_norm = F.normalize(wo)  # (bs, dc)  in_y (bs, nr)
        wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, self.label_size, 1)  # (bs, nr, dc)
        rel_emb = torch.mm(in_y, WL)  # (bs, dc)
        ay_emb = torch.mm(all_y, WL)  # (nr, dc)
        gt_dist = torch.norm(wo_norm - rel_emb, 2, 1)  # (bs, 1)
        all_dist = torch.norm(wo_norm_tile - ay_emb, 2, 2)  # (bs, nr)
        # print(all_dist)
        masking_y = torch.mul(in_y, 10000)
        # print(masking_y)
        _t_dist = torch.min(torch.add(all_dist, masking_y), 1)[0]
        # _t_dist=torch.min(all_dist,1)[0]
        # print(_t_dist.shape)
        loss = torch.mean(self.margin + gt_dist - _t_dist)
        return loss