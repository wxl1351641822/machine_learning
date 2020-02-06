from processData import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMforNer(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size,num_layers,batch_size):
        super(LSTMforNer, self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.tagset_size=tagset_size
        self.num_layers=num_layers
        self.batch_size=batch_size

        self.EMBEDDING=nn.Embedding(self.vocab_size,self.embedding_dim)
        self.LSTM=nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers=self.num_layers,batch_first=True,dropout=0.5,bidirectional=True)
       # 双向的结果是正向和反向拼接的结果。(官方的结果是不可复现的
       #  self.LSTM = nn.ModuleList()
       #  self.LSTM.append(nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim, \
       #                             num_layers=num_layers, batch_first=True, \
       #                             dropout=0.5, bidirectional=0))
       #  self.LSTM.append(nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim, \
       #                             num_layers=num_layers, batch_first=True, \
       #                             dropout=0.5, bidirectional=0))
        self.LINE=nn.Linear(self.hidden_dim*2,self.tagset_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers*2,self.batch_size,self.hidden_dim),
                 torch.zeros(self.num_layers*2,self.batch_size,self.hidden_dim))


    def forward(self,x,y):
        # print(torch.Tensor(x))
        length=[]
        max=0

        for i in range(len(x)):
            # embed[i]=self.EMBEDDING(torch.tensor(x[i]))
            if(max<len(x[i])):
                max=len(x[i])
            length.append(len(x[i]))
        length=torch.tensor(length)
        _, idx_sort = torch.sort(length, dim=0, descending=True)
        # print(idx_sort)
        length = list(length[idx_sort])  # 按下标取元素
        # print(idx_sort)
        idx=idx_sort.numpy().tolist()
        # x=[x[id]+[0]*(max-len(x[id])) for id in idx_sort]
        embed=torch.zeros(len(x),max,self.embedding_dim)
        # x_padding=[]
        oldy=[]
        for i in range(len(idx_sort)):
            # print(x[idx_sort[i]]+[0]*(max-len(x[idx_sort[i]])))
            embed[i]=self.EMBEDDING(torch.tensor(x[idx_sort[i]]+[0]*(max-len(x[idx_sort[i]]))))
            oldy.append(y[idx_sort[i]]+[0]*(max-len(x[idx_sort[i]])))
        embed = nn.utils.rnn.pack_padded_sequence(embed, length, batch_first=True)
        lstm_out,self.hidden=self.LSTM(embed,self.hidden)
        out, length = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # print(out)
        out = self.LINE(out)
        # print(out)
        tag_score=F.log_softmax(out,dim=2)
        # print(tag_score)
        # self.hidden=self.LSTM(embed.view(len(x),1,-1),self.hidden)
        result=torch.max(tag_score,axis=2)[1]
        # print(torch.tensor)
        oldy=torch.tensor(oldy)
        print(oldy.shape)
        acc=torch.zeros(len(length))
        for i in range(len(length)):
            model.zero_grad()
            # print(oldy[i,:length[i]])
            # print(result[i,:length[i]])
            result[i,length[i]:]=oldy[i,length[i]:]
            acc[i]=torch.sum(torch.eq(oldy[i,:length[i]],result[i,:length[i]])).float()/length[i]
        # print(acc)
        return tag_score,length,oldy,acc




traindata,dic_word_list,dic_label_list,dic_word,dic_label=getAllTrain()
# print(traindata)
EMBEDDING_DIM=300
HIDDEN_DIM=10
batch_size=100
model = LSTMforNer(EMBEDDING_DIM, HIDDEN_DIM, len(dic_word_list), len(dic_label_list),2,batch_size)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
with torch.no_grad():
    tag_score,length,oldy,acc=model(traindata[0][:100],traindata[1][:100])
    print(acc)
# model
batch_num=int(len(traindata[0])/batch_size)
for i in range(batch_num):
    tag_score,length,oldy,acc=model(traindata[0][i*batch_size:(i+1)*batch_size],traindata[1][i*batch_size:(i+1)*batch_size])
    # 第四步: 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度-
    # -多批次怎么求loss？？？
    for i in range(batch_size):
        loss = loss_function(tag_score[i][:length[i]], oldy[i][:length[i]])
    loss.backward()
    optimizer.step()
    print(acc)