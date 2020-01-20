from ner_model import *
import torch.optim as optim

dic_label2ids,dic_word2ids,train,ids2label=getData('CoNLL-2003/eng.train')
_,_,test,ids2label=getData('CoNLL-2003/eng.testa')
LOAD_MODEL=False
LOAD_MODEL_PARAMS='model/net_0_80_params.pkl'
LOAD_EPOCH=0
LOAD_BACHES=0

transition=tran(dic_label2ids,train)

model = BiLSTM_CRF(len(dic_word2ids), dic_label2ids, INPUT_SIZE, HIDDEN_SIZE,dic_label2ids,transition)
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)

torch.save(model, 'model/net.pkl')  # save entire net
# Check predictions before training
with torch.no_grad():
    # print(len(dic_word2ids))
    precheck_sent = prepare_sequence(train[1][0], dic_word2ids)
    precheck_tags = torch.tensor([dic_label2ids[t] for t in train[1][1]], dtype=torch.long)
    print(model(precheck_sent))

for epoch in range(EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data

    for i in range(len(train)//BATCH_size):
        acc_sum = 0
        if(LOAD_MODEL):
            LOAD_MODEL=False
            model.load_state_dict(torch.load(LOAD_MODEL_PARAMS))
        epoch=epoch+LOAD_EPOCH
        i=i+LOAD_BACHES
        for sentence, tags in train[(i*BATCH_size)%len(train):((i+1)*BATCH_size)%len(train)]:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            # print("step0")
            model.zero_grad()
            # print("step1")

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            # print(len(sentence))
            sentence_in = prepare_sequence(sentence, dic_word2ids)#从词的列表变成了id的列表
            targets = torch.tensor([dic_label2ids[t] for t in tags], dtype=torch.long)
            _,prediction=model(sentence_in)
            # acc=(prediction==targets).sum()

            acc_sum+=(targets==prediction).sum().float()/len(sentence_in)
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)#前向后向的loss
            # print("step2")
            # Step 4. Compute the loss, gradients, and update the parameters by

            loss.backward()
            # print("step3")
            optimizer.step()#用从loss得到的梯度更新
            # print("step4")
        acc=acc_sum/BATCH_size
        print("%d %d train_acc:"%(epoch,i),acc)
        acc_sum=0
        for sentence, tags in test[(i*BATCH_size//10)%len(test):(((i+1)*BATCH_size//10)%len(test))]:
            sentence_in = prepare_sequence(sentence, dic_word2ids)#从词的列表变成了id的列表
            targets = torch.tensor([dic_label2ids[t] for t in tags], dtype=torch.long)
            _,prediction=model(sentence_in)
            # acc=(prediction==targets).sum()

            acc_sum+=(targets==prediction).sum().float()/len(sentence_in)
        acc = acc_sum/BATCH_size*10
        print("%d %d test_acc:"%(epoch,i), acc)
        if(i%20==0):
            torch.save(model.state_dict(), 'model/net_%d_%d_params.pkl' % (epoch,i))
        # print((targets==prediction).sum(),len(sentence_in),(targets==prediction).sum()/len(sentence_in))

    torch.save(model.state_dict(), 'model/net_%d_params.pkl'%epoch)  # save only the parameters

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(train[1][0], dic_word2ids)
    print(model(precheck_sent))#这里得到预测结果