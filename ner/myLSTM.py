import torch
import torch.nn as nn
class LSTMTag:
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size,layers_num):
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.tagset_size=tagset_size
        self.layers_num=layers_num
        self.embedding_dim=embedding_dim
        self.embedding = torch.randn(self.vocab_size, self.embedding_dim)
        self.hidden=self.init_hidden()
        self.lstm=LSTM(self.embedding_dim,self.hidden_dim)
        self.lstm1=LSTM(self.hidden_dim,self.hidden_dim)
        self.hidden2tag=torch.randn(self.hidden_dim,self.tagset_size)
        self.lr=0.001

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers*num_directions, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
    def log_softmax(self,x):
        e=torch.exp(x)
        s=torch.sum(e,axis=1)
        return torch.log(e)-torch.log(s).reshape(x.shape[0],1);

    def forward(self,x):
        self.one_hot = torch.zeros(len(x), self.vocab_size).scatter_(1,x.reshape(len(x),1), 1)
        # embedding_matrix = torch.randn(self.vocab_size, self.embedding_dim)
        my_embed = torch.matmul(self.one_hot, self.embedding)
        self.o,(h,c)=self.lstm.single_layer_LSTM(my_embed.view(len(x), 1, -1),self.hidden)
        # o1,(h1,c1)=self.lstm1.single_layer_LSTM(o,self.hidden)
        # print(o1)
        tagspace=torch.matmul(self.o.view(len(x), -1),self.hidden2tag)
        # print(tagspace)
        self.tag_score=self.log_softmax(tagspace)
        # print(self.tag_score)
        # print(self.o.view(len(x),-1).shape)

        return self.tag_score

    def BP(self,y):
        one_hot_y = torch.zeros(len(y), self.tagset_size).scatter_(1, y.reshape(len(y), 1), -1.)
        self.Loss =one_hot_y*self.tag_score
        # print(self.Loss)
        dL_dtagspace=torch.exp(self.Loss)-1
        self.Loss=torch.sum(self.Loss,axis=1)
        # print(dL_dtagspace.shape)
        d_hidden2tag=torch.matmul(torch.transpose(self.o.view(len(x),-1),1,0),dL_dtagspace)
        dL_do=torch.matmul(dL_dtagspace,torch.transpose(self.hidden2tag,1,0))
        # print(dL_do)
        dL_dembedding=self.lstm.BPTT(dL_do.view(len(x),1,-1))
        # print(self.one_hot.shape)

        dL_dEm=torch.matmul(torch.transpose(self.one_hot,1,0),dL_dembedding.view(len(y),-1))
        # print(d_hidden2tag)
        self.hidden2tag=self.hidden2tag-d_hidden2tag*self.lr
        self.embedding-=dL_dEm*self.lr
        # print(self.hidden2tag)


        # d_hidden2tag=dL_dtagspace

class LSTM:
    def __init__(self,embedding_dim,hidden_dim):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        # 遗忘门
        self.Uf = torch.randn(self.embedding_dim, self.hidden_dim)
        self.Wf = torch.randn(self.hidden_dim, self.hidden_dim)
        self.bf = torch.randn(self.hidden_dim)

        #输入门
        self.Ug = torch.randn(self.embedding_dim, self.hidden_dim)
        self.Wg = torch.randn( self.hidden_dim, self.hidden_dim)
        self.bg = torch.randn( self.hidden_dim)

        # 状态
        self.Uc = torch.randn( self.embedding_dim, self.hidden_dim)
        self.Wc = torch.randn( self.hidden_dim, self.hidden_dim)
        self.bc = torch.randn( self.hidden_dim)

        # 输出门参数
        self.Uo = torch.randn(self.embedding_dim, self.hidden_dim)
        self.Wo = torch.randn(self.hidden_dim, self.hidden_dim)
        self.bo = torch.randn(self.hidden_dim)
        self.lr=0.001

    def sigmoid(self,x):
        return 1 / (1 + torch.exp(-1 * x))

    def tanh(self,x):
        return (torch.exp(x) - torch.exp(-1 * x)) / (torch.exp(x) + torch.exp(-1 * x))

    def LSTM_Cell(self,input_x, h0, c0):
        f1 = self.sigmoid(self.bf + torch.matmul(input_x, self.Uf) + torch.matmul(h0, self.Wf))
        #     print(f1)
        g1 =self.sigmoid(self.bg + torch.matmul(input_x, self.Ug) + torch.matmul(h0, self.Wg))
        #     print(g1)
        gc0=self.sigmoid(self.bc + torch.matmul(input_x, self.Uc) + torch.matmul(h0, self.Wc))
        c1 = f1 * c0 + g1 * gc0
        #     print(c1)
        q1 = self.sigmoid(self.bo + torch.matmul(input_x, self.Uo) + torch.matmul(h0, self.Wo))
        #     print(q1)
        h1 = self.tanh(c1) * q1
        #     print(h1)
        return (h1, c1),f1,g1,q1,gc0

    # forward
    def single_layer_LSTM(self,input_x,hidden):

        h0,c0=hidden
        self.h = torch.zeros(input_x.shape[0], input_x.shape[1], self.hidden_dim)
        self.c = torch.zeros(input_x.shape[0], input_x.shape[1], self.hidden_dim)
        self.f = torch.zeros(input_x.shape[0], input_x.shape[1], self.hidden_dim)
        self.g = torch.zeros(input_x.shape[0], input_x.shape[1], self.hidden_dim)
        self.q = torch.zeros(input_x.shape[0], input_x.shape[1], self.hidden_dim)
        self.x=input_x
        self.gc=torch.zeros(input_x.shape[0], input_x.shape[1], self.hidden_dim)
        for i in range(len(input_x)):
            (h0, c0),f0,g0,q0,gc0 = self.LSTM_Cell(input_x[i], h0, c0)
            self.h[i] = h0
            self.c[i] = c0
            self.f[i] = f0
            self.g[i] = g0
            self.q[i] = q0
            self.gc[i] = gc0
        return self.h, (h0, c0)

    def BPTT(self,dL_do):
        # dL_do=torch.cat((torch.zeros(1,dL_do.shape[1],dL_do.shape[2]),dL_do),axis=0)

        print(dL_do)
        dL_dq=torch.zeros(dL_do.shape)
        dL_ds=torch.zeros(dL_do.shape)
        dL_dqx = torch.zeros(dL_do.shape)

        # q
        dL_dbo = torch.zeros(self.bo.shape)
        h_t_1 = torch.zeros(self.h.shape)
        h_t_1[1:] = self.h[:-1]
        c_t_1 = torch.zeros(self.c.shape)
        c_t_1[1:] = self.c[:-1]
        dL_dWo = torch.zeros(self.Wo.shape)
        dL_dUo = torch.zeros(self.Uo.shape)

        # s
        dL_df=torch.zeros(dL_do.shape)
        dL_dg = torch.zeros(dL_do.shape)
        dL_dgcx = torch.zeros(dL_do.shape)


        #gc
        dL_dbc = torch.zeros(self.bc.shape)
        dL_dWc=torch.zeros(self.Wc.shape)
        dL_dUc=torch.zeros(self.Uc.shape)
        #f
        dL_dfx=torch.zeros(dL_do.shape)
        dL_dbf=torch.zeros(self.bf.shape)
        dL_dWf=torch.zeros(self.Wf.shape)
        dL_dUf=torch.zeros(self.Uf.shape)
        #g
        dL_dgx=torch.zeros(dL_do.shape)
        dL_dbg=torch.zeros(self.bg.shape)
        dL_dWg=torch.zeros(self.Wg.shape)
        dL_dUg=torch.zeros(self.Ug.shape)

        dL_dx = torch.zeros(self.x.shape)
        for i in range(len(dL_do)-1,-1,-1):
            print(i)
            dL_dq[i] = self.tanh(self.c[i]) * dL_do[i]
            dL_ds[i] += dL_do[i] * (1 - self.tanh(self.c[i]) * self.tanh(self.c[i])) * self.q[i]

            dL_dqx[i] = dL_dq[i] * self.q [i]* (1 - self.q[i])
            dL_dbo+=dL_dqx[i,0]
            dL_dWo += h_t_1[i, 0].reshape(self.hidden_dim, 1) * dL_dqx[i]
            # dL_dbo = dL_dqx
            dL_dUo += self.x[i].reshape(self.embedding_dim, 1) * dL_dqx[i]

            # s
            dL_df[i]=dL_ds[i]*c_t_1[i]
            dL_dg[i]=dL_ds[i]*self.gc[i]
            dL_dgcx[i]=dL_ds[i]*self.g[i]*self.gc[i]*(1-self.gc[i])

            #gc
            dL_dbc+=dL_dgcx[i,0]
            dL_dWc+=h_t_1[i, 0].reshape(self.hidden_dim, 1) * dL_dgcx[i]
            dL_dUc += self.x[i].reshape(self.embedding_dim, 1) * dL_dgcx[i]

            #f
            dL_dfx[i] = dL_df[i] * self.f[i] * (1 - self.f[i])
            dL_dbf += dL_dfx[i, 0]
            dL_dWf += h_t_1[i, 0].reshape(self.hidden_dim, 1) * dL_dfx[i]
            dL_dUf += self.x[i].reshape(self.embedding_dim, 1) * dL_dfx[i]

            # g
            dL_dgx[i] = dL_dg[i] * self.g[i] * (1 - self.g[i])
            dL_dbg += dL_dgx[i, 0]
            dL_dWg += h_t_1[i, 0].reshape(self.hidden_dim, 1) * dL_dgx[i]
            dL_dUg += self.x[i].reshape(self.embedding_dim, 1) * dL_dgx[i]



            if(i>1):
                dL_do[i-1]+=torch.matmul(dL_dqx[i],torch.transpose(self.Wo,1,0))
                dL_do[i - 1] += torch.matmul(dL_dgcx[i], torch.transpose(self.Wc, 1, 0))
                dL_do[i - 1] += torch.matmul(dL_dfx[i], torch.transpose(self.Wf, 1, 0))
                dL_do[i - 1] += torch.matmul(dL_dgx[i], torch.transpose(self.Wg, 1, 0))
                dL_ds[i-1]+=dL_ds[i]*self.f[i]


            dL_dx[i] += torch.matmul(dL_dqx[i], torch.transpose(self.Uo, 1, 0))
            # print(dL_dx)
            dL_dx[i] += torch.matmul(dL_dgcx[i], torch.transpose(self.Uc, 1, 0))
            # print(dL_dx)
            dL_dx[i] += torch.matmul(dL_dfx[i], torch.transpose(self.Uf, 1, 0))
            dL_dx[i] += torch.matmul(dL_dgx[i], torch.transpose(self.Ug, 1, 0))


        self.Wo-=self.lr*dL_dWo
        self.bo-=self.lr*dL_dbo
        self.Uo-=self.lr*dL_dUo
        self.Wc -= self.lr * dL_dWc
        self.bc-= self.lr * dL_dbc
        self.Uc -= self.lr * dL_dUc
        self.Wf -= self.lr * dL_dWf
        self.bf -= self.lr * dL_dbf
        self.Uf -= self.lr * dL_dUf
        self.Wg -= self.lr * dL_dWg
        self.bg -= self.lr * dL_dbg
        self.Ug -= self.lr * dL_dUg
        return dL_dx







        # dL_dqx=dL_dq * self.q * (1 - self.q)
        # dL_dbo = dL_dqx
        # h_t_1 = torch.zeros(self.h.shape)
        # h_t_1[1:] = self.h[:-1]
        # dL_dWo = torch.matmul(torch.transpose(h_t_1.view(self.h.shape[0], -1), 1, 0), dL_dbo.view(self.h.shape[0], -1))
        # dL_dUo = torch.matmul(torch.transpose(self.x.view(self.h.shape[0], -1), 1, 0), dL_dbo.view(self.h.shape[0], -1))
        # self.bo=self.bo-self.lr*dL_dbo
        # self.Wo=self.Wo-self.lr*dL_dWo

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# 实际中通常使用更大的维度如32维, 64维.
# 这里我们使用小的维度, 为了方便查看训练过程中权重的变化.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

model = LSTMTag(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),1)

x=prepare_sequence(training_data[0][0],word_to_ix)
y=prepare_sequence(training_data[0][1],tag_to_ix)
print(model.forward(x))
# model.BP(y)
for epoch in range(30):
    for sentence, tags in training_data:
        x = prepare_sequence(sentence, word_to_ix)
        y = prepare_sequence(tags, tag_to_ix)
        model.forward(x)
        model.BP(y)

x=prepare_sequence(training_data[0][0],word_to_ix)
y=prepare_sequence(training_data[0][1],tag_to_ix)
print(model.forward(x))