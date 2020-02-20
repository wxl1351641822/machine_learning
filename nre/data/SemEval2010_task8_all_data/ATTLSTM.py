import torch
import torch.nn as nn
import constant as constant
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm

class ATTLSTM(nn.Module):
    def __init__(self,vocab_size,label_size,embedding_size,hidden_size,pretrain_embedding_weight,dropout=0.7):
        super(ATTLSTM, self).__init__()
        self.vocab_size=vocab_size
        self.label_size=label_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.pretrain_embedding_weight=pretrain_embedding_weight
        self.dropout=dropout

        self.embed=nn.Embedding(vocab_size,embedding_size)
        self.embed.weight.data.copy_(torch.from_numpy(pretrain_embedding_weight))
        self.embed_drop=nn.Dropout(self.dropout)


        self.lstm=nn.LSTM(input_size=self.embedding_size,hidden_size=self.hidden_size,bidirectional=True,batch_first=True,num_layers=2,dropout=self.dropout)

        self.att = SelfAttention(self.hidden_size)

        self.fc=nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size,self.label_size),
            # nn.Tanh()
        )

    def init_hidden(self):
        return (torch.zeros(4, 1, self.hidden_size),
                torch.zeros(4, 1, self.hidden_size))

    def forward(self,inputs):
        # inputs（B,S,W)
        embedding=self.embed(inputs)
        # print(embedding.shape)
        #(B,S,E)
        # print(embedding)
        embedding_drop=self.embed_drop(embedding)
        # print(embedding_drop)
        lstm_out,(lstm_h,lstm_c)=self.lstm(embedding,self.init_hidden())
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
            nn.Linear(hidden_dim,1),

        )

    def forward(self,encoder_outputs):
        energy=self.projection(encoder_outputs)
        # print(energy.shape)
        weights=nn.functional.softmax(energy.squeeze(-1),dim=1)
        # print(weights.shape)
        # print(weights.unsqueeze(-1).shape)
        outputs=(encoder_outputs*weights.unsqueeze(-1)).sum(dim=1)
        return outputs,weights
datapath='./'
torch.manual_seed(41)
with open(datapath+constant.vocab_path,'r',encoding='utf-8') as f:
    vocab=eval(f.read())
    vocab_size=len(vocab)
with open(datapath+constant.label_path,'r',encoding='utf-8') as f:
    labellist=eval(f.read())
    label_size=len(labellist)
print(vocab_size)
print(label_size)

print('得到输入：')
train_data={}
test_data={}
embed_w=np.load(datapath+constant.emb_path)
model=ATTLSTM(vocab_size,label_size,constant.glove_dim,constant.lstm_hidden_dim,embed_w)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.01)
loss_func=nn.CrossEntropyLoss()
for epoch in range(50):
    print('epoch:', epoch)
    for i in range(10):
        print(datapath+constant.att_lstm_traindata_path+str(i))
        with open(datapath+constant.att_lstm_traindata_path+str(i),'r',encoding='utf-8') as f:
            train_data=eval(f.read())
        with open(datapath+constant.att_lstm_testdata_path+str(i),'r',encoding='utf-8') as f:
            test_data=eval(f.read())
        # print(train_data['token_id'])
        # train_data=Data.TensorDataset(torch.tensor(train_data['token_id'],dtype=torch.int64),torch.tensor(train_data['relation_id'],dtype=torch.int64))
        # print(train_data)
        # test_data=Data.TensorDataset(torch.tensor(test_data['token_id'],dtype=torch.int64),torch.tensor(test_data['relation_id'],dtype=torch.int64))
        # x,y=train_data['token_id'][0],train_data['relation_id'][0]
        # print(x,y)


        if epoch == 0:
            for p in optimizer.param_groups:
                p['lr'] = 0.005
        elif epoch == 2:
            for p in optimizer.param_groups:
                p['lr'] = 0.001
        elif epoch == 10:
            for p in optimizer.param_groups:
                p['lr'] = 0.0005
        elif  epoch  ==30:
            for p in optimizer.param_groups:
                p['lr'] = 0.0001
        optimizer.zero_grad()
        loss=torch.tensor(0.)
        acc=0.0
        for i,(x,y) in tqdm(enumerate(zip(train_data['token_id'],train_data['relation_id']))):
            pred=model(torch.tensor([x]))
            y=torch.tensor([y])
            loss+=loss_func(pred,y)
            soft = torch.log_softmax(pred, dim=1)
            predict = torch.argmax(soft, dim=1)
            if(predict==y[0]):
                acc+=1.0
                print(acc / 50., loss)
            # print(loss)

            if(i%16==0):

                acc = 0
                # print(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = torch.tensor(0.)

        with torch.no_grad():
            TP=torch.zeros(constant.label_size)
            FP = torch.zeros(constant.label_size)
            FN = torch.zeros(constant.label_size)
            for x,y in tqdm(zip(test_data['token_id'],test_data['relation_id'])):
                pred=model(torch.tensor([x]))
                y=torch.tensor(y)
                # print(pred)
                soft=torch.log_softmax(pred,dim=1)
                predict=torch.argmax(soft,dim=1)

                if(y==predict):
                    TP[y]+=1
                else:
                    FP[predict]+=1
                    FN[y]+=1
            print("acc:",TP.sum()/len(test_data['relation_id']))
            print(TP)
            print(FP)
            print(FN)
            P=TP/(TP+FP)
            R=TP/(TP+FN)
            F1=(2*P*R)/(P+R)
            print(F1)
            print(F1.sum())
