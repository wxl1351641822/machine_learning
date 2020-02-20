import torch
import torch.nn as nn
import constant
import numpy as np
# from CNN_nonline import ATTCNN as MODEL
from CNN_concate import ATTCNN as MODEL
# from data.SemEval2010_task8_all_data.ATTLSTM_with_position import ATTLSTM
# from ATTLSTM_with_batch import ATTLSTM
from tqdm import tqdm
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
model=MODEL(embed_w)
optimizer = torch.optim.Adam(model.parameters(), lr=constant.lr,weight_decay=constant.weight_decay)
lr=constant.lr
loss_func=nn.CrossEntropyLoss()
max_acc=torch.tensor(0.0)
# labellist=['cause-effect(e1,e2)', 'cause-effect(e2,e1)', 'component-whole(e1,e2)', 'component-whole(e2,e1)', 'content-container(e1,e2)', 'content-container(e2,e1)', 'entity-destination(e1,e2)', 'entity-destination(e2,e1)', 'entity-origin(e1,e2)', 'entity-origin(e2,e1)', 'instrument-agency(e1,e2)', 'instrument-agency(e2,e1)', 'member-collection(e1,e2)', 'member-collection(e2,e1)', 'message-topic(e1,e2)', 'message-topic(e2,e1)', 'other', 'product-producer(e1,e2)', 'product-producer(e2,e1)']
for epoch in range(constant.num_epoch):
    print('epoch:', epoch)


    for j in range(9):

        # print(datapath+constant.att_lstm_traindata_path+str(j))
        with open(datapath+constant.att_lstm_traindata_path+str(j),'r',encoding='utf-8') as f:
            train_data=eval(f.read())

        # print(train_data['token_id'])
        # train_data=Data.TensorDataset(torch.tensor(train_data['token_id'],dtype=torch.int64),torch.tensor(train_data['relation_id'],dtype=torch.int64))
        # print(train_data)
        # test_data=Data.TensorDataset(torch.tensor(test_data['token_id'],dtype=torch.int64),torch.tensor(test_data['relation_id'],dtype=torch.int64))
        # x,y=train_data['token_id'][0],train_data['relation_id'][0]
        # print(x,y)
        optimizer.zero_grad()
        acc=torch.tensor(0.0)
        batch_size = constant.batch_size
        batch_num=len(train_data['token_id'])//batch_size
        # predict=torch.tensor([0]*batch_size*batch_num)
        for i in tqdm(range(batch_num)):
            lr *= constant.lr_decay
            for p in optimizer.param_groups:
                p['lr'] = lr
            optimizer.zero_grad()
            x=train_data['token_id'][i*batch_size:(i+1)*batch_size]
            y=train_data['relation_id'][i*batch_size:(i+1)*batch_size]
            e1s=train_data['relation_subj_start'][i*batch_size:(i+1)*batch_size]
            e1e=train_data['relation_subj_end'][i*batch_size:(i+1)*batch_size]
            e2s=train_data['relation_obj_start'][i*batch_size:(i+1)*batch_size]
            e2e=train_data['relation_obj_end'][i*batch_size:(i+1)*batch_size]
            pos_1=train_data['position_1'][i*batch_size:(i+1)*batch_size]
            pos_2 = train_data['position_2'][i * batch_size:(i + 1) * batch_size]
        # for i,(x,y) in tqdm(enumerate(zip(train_data['token_id'],train_data['relation_id']))):
            pred=model(torch.tensor(x),e1s,e1e,e2s,e2e,pos_1,pos_2)
            y=torch.tensor(y)
            loss=loss_func(pred,y)
            # print(pos_1)
            # print(pos_2)


            # print(loss)

            # print(loss)
            loss.backward()
            optimizer.step()

        soft = torch.log_softmax(pred, dim=1)
        predict = torch.argmax(soft, dim=1)
        # print(predict)
        acc = torch.mean((predict == y).float())
        print(f'train {epoch}-{j}:acc={acc},loss={loss}')

    with open(datapath + constant.att_lstm_traindata_path + str(9), 'r', encoding='utf-8') as f:
        test_data = eval(f.read())
    with torch.no_grad():

        predict = torch.zeros( len(test_data['relation_id'])).long()
        for i in tqdm(range(len(test_data['token_id'])//batch_size)):
            optimizer.zero_grad()
            x = test_data['token_id'][i * batch_size:(i + 1) * batch_size]
            y = test_data['relation_id'][i * batch_size:(i + 1) * batch_size]
            e1s = test_data['relation_subj_start'][i * batch_size:(i + 1) * batch_size]
            e1e = test_data['relation_subj_end'][i * batch_size:(i + 1) * batch_size]
            e2s = test_data['relation_obj_start'][i * batch_size:(i + 1) * batch_size]
            e2e = test_data['relation_obj_end'][i * batch_size:(i + 1) * batch_size]
            pos_1 = test_data['position_1'][i * batch_size:(i + 1) * batch_size]
            pos_2 = test_data['position_2'][i * batch_size:(i + 1) * batch_size]
            # for i,(x,y) in tqdm(enumerate(zip(train_data['token_id'],train_data['relation_id']))):

            pred = model(torch.tensor(x), e1s, e1e, e2s, e2e, pos_1, pos_2)
            # print(pred)
            soft=torch.log_softmax(pred,dim=1)
            predict[i * batch_size:(i + 1) * batch_size]=torch.argmax(soft,dim=1)
        s1=''
        s2=''
        for id,r_id,p_id in zip(test_data['id'],test_data['relation_id'],predict):
            s1+=str(id)+'	'+labellist[r_id]+'\n'
            s2 += str(id) +'	'+ labellist[p_id] + '\n'
        with open('./result/answer'+str(epoch),'w',encoding='utf-8') as f:
            f.write(s1)
        with open('./result/proposed'+str(epoch),'w',encoding='utf-8') as f:
            f.write(s2)
        y=torch.tensor(test_data['relation_id'])

        TP=torch.zeros(label_size)
        FP = torch.zeros(label_size)
        FN = torch.zeros(label_size)
        for i in range(y.shape[0]):
            # print(y[i],predict[i])
            if(y[i]==predict[i]):
                TP[y[i]]+=1
            else:
                FP[predict[i]]+=1
                FN[y[i]]+=1
        print(TP)
        print(FP)
        print(FN)
        P=TP.float()/(TP.float()+FP.float())
        R=TP.float()/(TP.float()+FN.float())
        F1=(2*P*R)/(P+R)
        print(F1)
        acc=torch.mean((predict==y).float())
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), constant.checkpoint_path)
        print(f'--------------------------------------------test:acc={acc},macro-F1={F1.sum()},max_acc={max_acc}-------------------------------------------')
