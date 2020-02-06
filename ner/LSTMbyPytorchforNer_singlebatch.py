import torch
import torch.nn as nn
import torch.nn.functional as F
from processData import *
from tqdm import tqdm

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers=2,dropout=0.5,bidirectional=True)

        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers*num_directions, batch_size, hidden_dim)
        return (torch.zeros(4, 1, self.hidden_dim),
                torch.zeros(4, 1, self.hidden_dim))

    def forward(self, sentence):
        #         embeds=sentence

        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        # print(lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def measure(predict,y):
    acc = (torch.sum(torch.eq(predict, y))).type(torch.FloatTensor) / float(len(y))
    TP=torch.zeros( len(dic_label_list),dtype=float)
    FP=torch.zeros( len(dic_label_list),dtype=float)
    FN=torch.zeros( len(dic_label_list),dtype=float)
    for i in range(len(y)):
        if(y[i]==predict[i]):
            TP[y[i]]+=1
        else:
            FP[predict[i]]+=1
            FN[y[i]]+=1
    # micro:算总的
    # print(torch.sum(TP))
    micro_precision=torch.sum(TP)/(torch.sum(TP)+torch.sum(FP))
    micro_recall=torch.sum(TP)/(torch.sum(TP)+torch.sum(FN))
    micro_F1=2*(micro_precision*micro_recall)/(micro_precision+micro_recall)
    # macro ：算每一类的然后平均
    TP[TP==0]=1e-8
    FP[FP==0]=1e-8
    FN[FN==0]=1e-8
    macro_precision=TP/(TP+FP)
    macro_recall=TP/(TP+FN)

    macro_F1=2*(macro_recall*macro_precision)/(macro_recall+macro_precision)
    print(macro_F1)
    macro_F1=torch.mean(macro_F1)
    print(acc,micro_F1,macro_F1)
    return acc,micro_F1,macro_F1

traindata,dic_word_list,dic_label_list,dic_word,dic_label=getAllTrain()
# print(traindata)
EMBEDDING_DIM=300
HIDDEN_DIM=10
batch_size=100
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,len(dic_word_list), len(dic_label_list))
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 查看训练前的分数
# 注意: 输出的 i,j 元素的值表示单词 i 的 j 标签的得分
# 这里我们不需要训练不需要求导所以使用torch.no_grad()
with torch.no_grad():
    inputs = torch.tensor(traindata[0][1])
    tag_scores = model(inputs)
    # print(tag_scores)

for epoch in range(2):  # 实际情况下你不会训练300个周期, 此例中我们只是随便设了一个值
    for sentence, tags in tqdm(zip(traindata[0][:-100],traindata[1][:-100])):
        # 第一步: 请记住Pytorch会累加梯度.
        # 我们需要在训练每个实例前清空梯度
        model.zero_grad()

        # 此外还需要清空 LSTM 的隐状态,
        # 将其从上个实例的历史中分离出来.
        model.hidden = model.init_hidden()

        # 准备网络输入, 将其变为词索引的 Tensor 类型数据
        sentence_in = torch.tensor(sentence)
        targets = torch.tensor(tags)


        # 第三步: 前向传播.
        tag_scores = model(sentence_in)
        predict =torch.max(tag_scores,axis=1)[1]
        # predict = torch.max(tag_scores, axis=1)[1]
        # measure(predict, targets)
        # 第四步: 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()



# 查看训练后的得分
with torch.no_grad():
    y = torch.tensor([traindata[1][-100]])
    sentence_in = torch.tensor(traindata[0][-100])
    tag_scores = model(sentence_in)

    predict = torch.max(tag_scores, axis=1)[1].reshape(1,len(traindata[1][-100]))

    for sentence, tags in zip(traindata[0][-101:], traindata[1][-101:]):
        # 准备网络输入, 将其变为词索引的 Tensor 类型数据
        sentence_in = torch.tensor(sentence)
        # targets = torch.tensor(tags)
        tag_scores = model(sentence_in)

        predict =torch.cat((predict,torch.max(tag_scores,axis=1)[1].reshape(1,len(tags))),axis=1)
        y = torch.cat((y, torch.tensor([tags])), axis=1)


        x0=[dic_word_list[s] for s in sentence]
        y0=[dic_label_list[t] for t in tags]
        print(x0)
        print(y0)

    # print(predict.shape)
    # print(y.shape)
    measure(predict.reshape(y.shape[1]),y.reshape(y.shape[1]))
    # 句子是 "the dog ate the apple", i,j 表示对于单词 i, 标签 j 的得分.
    # 我们采用得分最高的标签作为预测的标签. 从下面的输出我们可以看到, 预测得
    # 到的结果是0 1 2 0 1. 因为 索引是从0开始的, 因此第一个值0表示第一行的
    # 最大值, 第二个值1表示第二行的最大值, 以此类推. 所以最后的结果是 DET
    # NOUN VERB DET NOUN, 整个序列都是正确的!
    # print(tag_scores)