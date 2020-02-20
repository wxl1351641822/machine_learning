import torch
import torch.nn as nn
import constant as constant
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm

class ATTLSTM(nn.Module):
    def __init__(self,vocab_size,label_size,embedding_size,hidden_size,pretrain_embedding_weight,dropout=0.5):
        super(ATTLSTM, self).__init__()
        self.vocab_size=vocab_size
        self.label_size=label_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.pretrain_embedding_weight=pretrain_embedding_weight
        self.dropout=dropout
        self.W = nn.Parameter(torch.randn(200, self.label_size, 200))

        self.embed=nn.Embedding(vocab_size,embedding_size)
        self.embed.weight.data.copy_(torch.from_numpy(pretrain_embedding_weight))
        self.embed_drop=nn.Dropout(self.dropout)


        self.lstm=nn.LSTM(input_size=self.embedding_size,hidden_size=self.hidden_size,bidirectional=True,batch_first=True,num_layers=2,dropout=self.dropout)

        # self.att = SelfAttention(self.hidden_size)
        self.distance_embedding=nn.Embedding(70,20)
        self.fc=nn.Sequential(
            nn.Linear(400,self.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size,self.label_size),
            # nn.Tanh()
        )


    def init_hidden(self):
        return (torch.zeros(4, 1, self.hidden_size),
                torch.zeros(4, 1, self.hidden_size))

    def forward(self,inputs,e1s,e1e,e2s,e2e,p1,p2):
        # inputs（B,S,W)
        embedding=self.embed(torch.tensor([inputs]))
        # print(embedding.shape)
        #(B,S,E)
        # print(embedding)
        embedding_drop=self.embed_drop(embedding)
        # print(embedding_drop)
        lstm_out,(lstm_h,lstm_c)=self.lstm(embedding,self.init_hidden())
        # print(lstm_out.shape)
        # lstm_out:(B,S,H)
        feature=lstm_out[:,:,:self.hidden_size]+lstm_out[:,:,self.hidden_size:]
        p1_distance = self.distance_embedding(torch.tensor([p1]))
        p2_distance = self.distance_embedding(torch.tensor([p2]))
        # print(feature.shape)
        m1=torch.mean(feature[:,e1s:e1e+1],axis=1)
        E_d21=torch.mean(p2_distance[:,e1s:e1e+1],axis=1)
        # e1 = torch.cat([m1, E_d21], 1)
        # print(m1.shape)
        # print(e2s,e2e)
        # print(feature[:,e2s:e2e+1].shape)
        m2=torch.mean(feature[:,e2s:e2e+1],axis=1)
        E_d12 = torch.mean(p1_distance[:, e2s:e2e + 1], axis=1)
        e=torch.cat([m1, E_d21,m2,E_d12],1)
        #
        # torch.matmul(torch.matmul(e1,self.W),e2)

        # print(att_out.shape)
        out=self.fc(e)
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
datapath='./'
torch.manual_seed(41)
# with open(datapath+constant.vocab_path,'r',encoding='utf-8') as f:
#     vocab=eval(f.read())
#     vocab_size=len(vocab)
# with open(datapath+constant.label_path,'r',encoding='utf-8') as f:
#     labellist=eval(f.read())
#     label_size=len(labellist)
vocab_size=22731
label_size=19

print('得到输入：')
train_data={}
test_data={}
with open(datapath+constant.att_lstm_traindata_path) as f:
    train_data=eval(f.read())
with open(datapath+constant.att_lstm_testdata_path) as f:
    test_data=eval(f.read())
# print(train_data['token_id'])
# train_data=Data.TensorDataset(torch.tensor(train_data['token_id'],dtype=torch.int64),torch.tensor(train_data['relation_id'],dtype=torch.int64))
# print(train_data)
# test_data=Data.TensorDataset(torch.tensor(test_data['token_id'],dtype=torch.int64),torch.tensor(test_data['relation_id'],dtype=torch.int64))
embed_w=np.load(datapath+constant.emb_path)
model=ATTLSTM(vocab_size,label_size,constant.glove_dim,constant.lstm_hidden_dim,embed_w)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.001)
loss_func=nn.CrossEntropyLoss()
# train_data = Data.TensorDataset(torch.tensor(train_data['token_id'], dtype=torch.int64), torch.tensor(train_data['relation_id'], dtype=torch.int64),
#                                 torch.tensor(train_data['relation_subj_start'], dtype=torch.int64), torch.tensor(train_data['relation_subj_end'], dtype=torch.int64),
#                                 torch.tensor(train_data['relation_obj_start'], dtype=torch.int64), torch.tensor(train_data['relation_obj_end'], dtype=torch.int64),
#                                 torch.tensor(train_data['position_1'], dtype=torch.int64),torch.tensor(train_data['position_2'], dtype=torch.int64))
# train_data_loader = Data.DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=1)
# x,y=train_data['token_id'][0],train_data['relation_id'][0]
# print(x,y)
for epoch in range(50):
    print('epoch:',epoch)
    if epoch == 0:
        for p in optimizer.param_groups:
            p['lr'] = 0.005
    elif epoch == 4:
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
    acc=0.
    for i,(x,y,e1s,e1e,e2s,e2e,p1,p2) in tqdm(enumerate(zip(train_data['token_id'],train_data['relation_id'],train_data['relation_subj_start'], train_data['relation_subj_end'],train_data['relation_obj_start'], train_data['relation_obj_end'],train_data['position_1'],train_data['position_2']))):
    # for i, (x, y, e1s, e1e, e2s, e2e, p1, p2) in tqdm(enumerate(train_data_loader)):
        pred=model(x,e1s,e1e,e2s,e2e,p1,p2)
        # loss+=loss_func(pred,torch.tensor([y]))
        y = torch.tensor([y])
        loss += loss_func(pred, y)
        soft = torch.log_softmax(pred, dim=1)
        predict = torch.argmax(soft, dim=1)
        if (predict == y[0]):
            acc += 1.0
        # print(loss)
        if (i % 1600 == 0):
            print(acc / 1600., loss)
            acc = 0.0

        if(i%16==0):
            # print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = torch.tensor(0.)

    with torch.no_grad():
        TP=torch.zeros(constant.label_size)
        FP = torch.zeros(constant.label_size)
        FN = torch.zeros(constant.label_size)
        for x,y,e1s,e1e,e2s,e2e,p1,p2 in tqdm(zip(test_data['token_id'],test_data['relation_id'],test_data['relation_subj_start'], test_data['relation_subj_end'],test_data['relation_obj_start'], test_data['relation_obj_end'],test_data['position_1'],test_data['position_2'])):
            pred=model(x,e1s,e1e,e2s,e2e,p1,p2)
            y=torch.tensor(y)
            # print(pred)
            soft=torch.log_softmax(pred,dim=1)
            predict=torch.argmax(soft,dim=1)
            # print(predict)
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
