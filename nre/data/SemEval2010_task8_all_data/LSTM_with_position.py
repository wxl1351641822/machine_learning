import torch
import torch.nn as nn
import constant as constant
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm

class ATTLSTM(nn.Module):
    def __init__(self,vocab_size,label_size,embedding_size,hidden_size,pretrain_embedding_weight,batch_size,dropout=0.7):
        super(ATTLSTM, self).__init__()
        self.vocab_size=vocab_size
        self.label_size=label_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.pretrain_embedding_weight=pretrain_embedding_weight
        self.dropout=dropout

        self.batch_size=batch_size

        self.embed=nn.Embedding(vocab_size,embedding_size)
        self.embed.weight.data.copy_(torch.from_numpy(pretrain_embedding_weight))
        self.embed_drop=nn.Dropout(self.dropout)
        self.distance1_embedding = nn.Embedding(constant.max_length*2, constant.pos_embedding_dim)
        self.distance2_embedding = nn.Embedding(constant.max_length * 2, constant.pos_embedding_dim)

        self.lstm=nn.LSTM(input_size=self.embedding_size+20,hidden_size=self.hidden_size,bidirectional=True,batch_first=True,num_layers=2,dropout=self.dropout)

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
        word_embed=self.embed(inputs)
        p1=torch.tensor(p1)
        # print(p2)
        p2=torch.tensor(p2)
        # print(p2.shape)
        p1_distance = self.distance1_embedding(p1)
        # print(p2.shape)
        p2_distance = self.distance2_embedding(p2)
        embedding=torch.cat([word_embed,p1_distance,p2_distance],-1)
        # print(embedding.shape)
        #(B,S,E)
        # print(embedding)
        embedding_drop=self.embed_drop(embedding)
        # print(embedding_drop)
        lstm_out,(lstm_h,lstm_c)=self.lstm(embedding_drop,self.init_hidden())
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
        #
        )

    def forward(self,encoder_outputs):
        energy=self.projection(encoder_outputs)
        # print(energy.shape)
        weights=nn.functional.softmax(energy.squeeze(-1),dim=1)
        # print(weights.shape)
        # print(weights.unsqueeze(-1).shape)
        outputs=torch.tanh(encoder_outputs*weights.unsqueeze(-1)).sum(dim=1)
        return outputs,weights
# datapath='./'
# torch.manual_seed(1234)
# with open(datapath+constant.vocab_path,'r',encoding='utf-8') as f:
#     vocab=eval(f.read())
#     vocab_size=len(vocab)
# with open(datapath+constant.label_path,'r',encoding='utf-8') as f:
#     labellist=eval(f.read())
#     label_size=len(labellist)
# print(vocab_size)
# print(label_size)
#
# print('得到输入：')
# train_data={}
# test_data={}
# embed_w=np.load(datapath+constant.emb_path)
# model=ATTLSTM(vocab_size,label_size,constant.glove_dim,constant.lstm_hidden_dim,embed_w,batch_size=16,dropout=0.7)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_func=nn.CrossEntropyLoss()
# for epoch in range(50):
#     print('epoch:', epoch)
#     if epoch == 0:
#         for p in optimizer.param_groups:
#             p['lr'] = 0.005
#     elif epoch == 2:
#         for p in optimizer.param_groups:
#             p['lr'] = 0.001
#     elif epoch == 10:
#         for p in optimizer.param_groups:
#             p['lr'] = 0.0005
#     elif epoch == 30:
#         for p in optimizer.param_groups:
#             p['lr'] = 0.0001
#     for j in range(9):
#         # print(datapath+constant.att_lstm_traindata_path+str(j))
#         with open(datapath+constant.att_lstm_traindata_path+str(j),'r',encoding='utf-8') as f:
#             train_data=eval(f.read())
#
#         # print(train_data['token_id'])
#         # train_data=Data.TensorDataset(torch.tensor(train_data['token_id'],dtype=torch.int64),torch.tensor(train_data['relation_id'],dtype=torch.int64))
#         # print(train_data)
#         # test_data=Data.TensorDataset(torch.tensor(test_data['token_id'],dtype=torch.int64),torch.tensor(test_data['relation_id'],dtype=torch.int64))
#         # x,y=train_data['token_id'][0],train_data['relation_id'][0]
#         # print(x,y)
#         optimizer.zero_grad()
#         acc=torch.tensor(0.0)
#         batch_size = 16
#         batch_num=len(train_data['token_id'])//batch_size
#         # predict=torch.tensor([0]*batch_size*batch_num)
#         for i in tqdm(range(batch_num)):
#             optimizer.zero_grad()
#             x=train_data['token_id'][i*batch_size:(i+1)*batch_size]
#             y=train_data['relation_id'][i*batch_size:(i+1)*batch_size]
#             e1s=train_data['relation_subj_start'][i*batch_size:(i+1)*batch_size]
#             e1e=train_data['relation_subj_end'][i*batch_size:(i+1)*batch_size]
#             e2s=train_data['relation_obj_start'][i*batch_size:(i+1)*batch_size]
#             e2e=train_data['relation_obj_end'][i*batch_size:(i+1)*batch_size]
#             pos_1=train_data['position_1'][i*batch_size:(i+1)*batch_size]
#             pos_2 = train_data['position_2'][i * batch_size:(i + 1) * batch_size]
#         # for i,(x,y) in tqdm(enumerate(zip(train_data['token_id'],train_data['relation_id']))):
#             pred=model(torch.tensor(x),e1s,e1e,e2s,e2e,pos_1,pos_2)
#             y=torch.tensor(y)
#             loss=loss_func(pred,y)
#
#
#
#             # print(loss)
#
#             # print(loss)
#             loss.backward()
#             optimizer.step()
#
#         soft = torch.log_softmax(pred, dim=1)
#         predict = torch.argmax(soft, dim=1)
#         # print(predict)
#         acc = torch.mean((predict == y).float())
#         print(f'train {epoch}-{j}:acc={acc},loss={loss}')
#
#     with open(datapath + constant.att_lstm_traindata_path + str(9), 'r', encoding='utf-8') as f:
#         test_data = eval(f.read())
#     with torch.no_grad():
#         TP=torch.zeros(constant.label_size)
#         FP = torch.zeros(constant.label_size)
#         FN = torch.zeros(constant.label_size)
#         predict = torch.zeros( len(test_data['relation_id'])).long()
#         for i in tqdm(range(len(test_data['token_id'])//batch_size)):
#             optimizer.zero_grad()
#             x = test_data['token_id'][i * batch_size:(i + 1) * batch_size]
#             y = test_data['relation_id'][i * batch_size:(i + 1) * batch_size]
#             e1s = test_data['relation_subj_start'][i * batch_size:(i + 1) * batch_size]
#             e1e = test_data['relation_subj_end'][i * batch_size:(i + 1) * batch_size]
#             e2s = test_data['relation_obj_start'][i * batch_size:(i + 1) * batch_size]
#             e2e = test_data['relation_obj_end'][i * batch_size:(i + 1) * batch_size]
#             pos_1 = test_data['position_1'][i * batch_size:(i + 1) * batch_size]
#             pos_2 = test_data['position_2'][i * batch_size:(i + 1) * batch_size]
#             # for i,(x,y) in tqdm(enumerate(zip(train_data['token_id'],train_data['relation_id']))):
#             pred = model(torch.tensor(x), e1s, e1e, e2s, e2e, pos_1, pos_2)
#             # print(pred)
#             soft=torch.log_softmax(pred,dim=1)
#             predict[i * batch_size:(i + 1) * batch_size]=torch.argmax(soft,dim=1)
#         s1 = ''
#         s2 = ''
#         for id, r_id, p_id in zip(test_data['id'], test_data['relation_id'], predict):
#             s1+=str(id)+'	'+labellist[r_id]+'\n'
#             s2 += str(id) +'	'+ labellist[p_id] + '\n'
#         with open('./result/answer' +str(epoch), 'w', encoding='utf-8') as f:
#             f.write(s1)
#         with open('./result/proposed' +str(epoch), 'w', encoding='utf-8') as f:
#             f.write(s2)
#
#         y=torch.tensor(test_data['relation_id'])
#
#         TP=torch.zeros(label_size)
#         FP = torch.zeros(label_size)
#         FN = torch.zeros(label_size)
#         for i in range(y.shape[0]):
#             # print(y[i],predict[i])
#             if(y[i]==predict[i]):
#                 TP[y[i]]+=1
#             else:
#                 FP[predict[i]]+=1
#                 FN[y[i]]+=1
#         print(TP)
#         print(FP)
#         print(FN)
#         P=TP.float()/(TP.float()+FP.float())
#         R=TP.float()/(TP.float()+FN.float())
#         F1=(2*P*R)/(P+R)
#         print(F1)
#         print(f'--------------------------------------------test:acc={torch.mean((predict==y).float())},macro-F1={F1.sum()}-------------------------------------------')
