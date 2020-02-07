import torch
import torch.nn as nn
import torch.nn.functional as F
from processData import *
from tqdm import tqdm
import torch.optim as optim
torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tagset,embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.tagset_size=len(tagset)


        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,num_layers=2,dropout=0.5,bidirectional=True)
        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)


        #CRF的参数
        self.A=nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))#行为t,列为t-1

        self.A.data[0,:]=-1000
        self.A.data[:,self.tagset_size-1]=-1000
        # print(self.A)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers*num_directions, batch_size, hidden_dim)
        return (torch.zeros(4, 1, self.hidden_dim//2),
                torch.zeros(4, 1, self.hidden_dim//2))

    def lstm_forward(self, sentence):
        #         embeds=sentence

        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        # print(lstm_out)
        lstm_feats = torch.tanh(self.hidden2tag(lstm_out.view(len(sentence), -1)))
        return lstm_feats


        # 解码
    def Viterbi_M(self,features):
        sequence_len=features.shape[0]
        delta = torch.full((1, self.tagset_size), -1000.)
        delta[0][0]=0;
        # logM = torch.log(features)
        forward=[]
        forward.append(delta)
        # print(logM[0,0].shape)
        # print(delta.shape)
        # print(delta[0])
        # delta[0] = logM[0]
        # torch.max(delta[0].reshape(self.y_size, 1) + logM[1], axis=0)
        indices = []
        for i in range(len(features)):
            gamma_r_l=forward[i]
            # print(gamma_r_l+self.A)
            delta,indice=torch.max(gamma_r_l+self.A,dim=1)
            # print(delta)
            # print(indice)
            delta=features[i]+delta
            forward.append(delta.reshape(1,self.tagset_size))
            indices.append(indice.tolist())
        terminal = forward[-1] + self.A[self.tagset_size - 1]
        best_tag_id = torch.argmax(terminal).tolist()
        best_score = terminal[0][best_tag_id]
        #     print(best_tag_id)
        #     print(best_score)
        bestpath = [best_tag_id]
        for indice in reversed(indices):
            best_tag_id = indice[best_tag_id]
            bestpath.append(best_tag_id)
        bestpath.pop()
        bestpath.reverse()
        return bestpath, best_score
            # print(gamma_r_l.shape)
            # print(delta[i - 1].reshape(self.y_size, 1) + logM[i])
        #     delta[i], indice = torch.max(delta[i - 1].reshape(self.tagset_size, 1) + logM[i], axis=0)
        #     indices.append(indice)
        # # print(delta)
        # #     print(indices)
        # path = torch.zeros(sequence_len, dtype=torch.long)
        # #     print(path)
        # path[sequence_len - 1] = torch.argmax(delta[sequence_len - 1])
        # #     print(path)
        # for i in range(sequence_len - 2, -1, -1):
        #     path[i] = indices[i][path[i + 1]]
        # #     print(path)
        # return path

    def forward(self,sentence):
        lstm_feats=self.lstm_forward(sentence)#len(sentence),self.tagset_size
        # print(lstm_feats.shape)
        path,score=self.Viterbi_M(lstm_feats)
        # print(path,score)
        return path,score

    def alpha_alg(self,feats):
        init_alpha = torch.full([self.tagset_size], -1000.)
        init_alpha[0] = 0.
        alpha = [init_alpha]
        for i in range(feats.shape[0]):
            gamma = alpha[i]
            aa = gamma + self.A + feats[i].reshape(self.tagset_size, 1)
            alpha.append(torch.logsumexp(aa, axis=1))
        terminal = alpha[-1] + self.A[self.tagset_size - 1]
        logZ = torch.logsumexp(terminal, axis=0)
        return logZ;

    def gold_score(self,feats, y):
        goldScore = self.A[y[0], 0] + feats[0, y[0]]
        for i in range(len(y) - 1):
            goldScore += self.A[y[i + 1], y[i]] + feats[i + 1, y[i + 1]]
        return goldScore

    def neg_log_likelihood(self,sentence,labels):
        lstm_feats=self.lstm_forward(sentence)
        logZ=self.alpha_alg(lstm_feats)
        gold_score=self.gold_score(lstm_feats,labels)
        return logZ-gold_score

def measure(predict,y):
    acc = (torch.sum(torch.eq(predict, y))).type(torch.FloatTensor) / float(len(y))
    TP=torch.zeros( 7,dtype=float)
    FP=torch.zeros( 7,dtype=float)
    FN=torch.zeros( 7,dtype=float)
    for i in range(len(y)):
        if(y[i]==predict[i]):
            TP[y[i]-1]+=1
        else:
            FP[predict[i]-1]+=1
            FN[y[i]-1]+=1
    # micro:算总的
    # print(torch.sum(TP))
    print(TP)
    print(FP)
    print(FN)
    micro_precision=torch.sum(TP)/(torch.sum(TP)+torch.sum(FP))
    micro_recall=torch.sum(TP)/(torch.sum(TP)+torch.sum(FN))
    micro_F1=2*(micro_precision*micro_recall)/(micro_precision+micro_recall)
    # macro ：算每一类的然后平均
    # TP[TP==0]=1e-8
    # FP[FP==0]=1e-8
    # FN[FN==0]=1e-8
    macro_precision=TP/(TP+FP)
    macro_recall=TP/(TP+FN)

    macro_F1=2*(macro_recall*macro_precision)/(macro_recall+macro_precision)
    print(macro_F1)
    macro_F1=torch.mean(macro_F1)
    print(acc,micro_F1,macro_F1)
    return acc,micro_F1,macro_F1

if __name__== '__main__':
    START_TAG = "<BEG>"
    STOP_TAG = "<END>"
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    training_data, dic_word_list, dic_label_list, word_to_ix, tag_to_ix = getAllTrain()

    # Make up some training data
    # training_data = [(
    #     "the wall street journal reported today that apple corporation made money".split(),
    #     "B I I I O O O B I O O".split()
    # ), (
    #     "georgia tech is a university in georgia".split(),
    #     "B I O O O O B".split()
    # )]
    #
    # word_to_ix = {}
    # for sentence, tags in training_data:
    #     for word in sentence:
    #         if word not in word_to_ix:
    #             word_to_ix[word] = len(word_to_ix)
    #
    # tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    #
    model = BiLSTM_CRF(len(dic_word_list), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    # print( list(model.named_parameters()))
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    torch.save(model, 'model/net1.pkl')
    # Check predictions before training
    with torch.no_grad():
        precheck_sent = torch.tensor(training_data[0][0])
        # print(precheck_sent)
        precheck_tags = torch.tensor(training_data[1][0])
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in tqdm(zip(training_data[0][:-2000],training_data[1][:-2000])):
            # print(sentence,tags)
            # Step 1. Remmber that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            sentence_in=torch.tensor(sentence)
            targets=torch.tensor(tags)
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix)
            # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            score,predict=model(sentence_in)
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward(retain_graph=True)
            optimizer.step()
        torch.save(model.state_dict(), 'model/net1_%d_params.pkl' % epoch)

    # Check predictions after training
    with torch.no_grad():
        # precheck_sent = torch.tensor(training_data[0][0])
        # print(model(precheck_sent))
        y = torch.tensor([training_data[1][-2000]])
        sentence_in = torch.tensor(training_data[0][-2000])
        tag_scores,predict1 = model(sentence_in)

        predict = torch.tensor([predict1])

        for sentence, tags in tqdm(zip(training_data[0][:-2000], training_data[1][:-2000])):
            # 准备网络输入, 将其变为词索引的 Tensor 类型数据
            sentence_in = torch.tensor(sentence)
            # targets = torch.tensor(tags)
            tag_scores,predict1 = model(sentence_in)
            predict = torch.cat((predict, torch.tensor([predict1])), axis=1)
            y = torch.cat((y, torch.tensor([tags])), axis=1)

            x0 = [dic_word_list[s] for s in sentence]
            y0 = [dic_label_list[t] for t in tags]
            predict0 = [dic_label_list[t] for t in predict1]
            print(x0)
            print(y0)
            print(predict0)
        # print(predict.shape)
        # print(y.shape)
        measure(predict.reshape(y.shape[1]), y.reshape(y.shape[1]))