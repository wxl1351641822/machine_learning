import torch
import torch.nn as nn
from processData import *
from tqdm import tqdm
import re


# https://www.cnblogs.com/pinard/p/7048333.html

def get_feature_functions(word_sets, labels, observes):
    """生成各种特征函数"""
    print("get feature functions ...")
    transition_functions = [
        lambda yp, y, x_v, i, _yp=_yp, _y=_y: 1 if yp == _yp and y == _y else 0
        for _yp in labels[:-1] for _y in labels[1:]
    ]

    # print(labels)

    def set_membership(tag, word_sets):
        def fun(yp, y, x_v, i):
            if i < len(x_v) and x_v[i] in word_sets[tag]:
                return 1
            else:
                return 0

        return fun

    observation_functions = [set_membership(t, word_sets) for t in range(len(word_sets))]

    # misc_functions = [
    #     lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[^0-9a-zA-Z]+$', x_v[i]) else 0,
    #     lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[A-Z\.]+$', x_v[i]) else 0,
    #     lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[0-9\.]+$', x_v[i]) else 0
    # ]

    # tagval_functions = [
    #     lambda yp, y, x_v, i, _y=_y, _x=_x: 1 if i < len(x_v) and y == _y and x_v[i] == _x else 0
    #     for _y in labels
    #     for _x in observes]

    return transition_functions + observation_functions


class CRF(nn.Module):
    def __init__(self, x_size, y_size, func):
        self.sequence_len=0;
        self.x_size = x_size;  # 字的词典长度
        self.y_size = y_size;
        self.k = len(func)
        self.func = func
        self.w = torch.rand(self.k)
        self.f=0
        self.M=0;
        self.epsilon=1e-8
        # 转移
        # 每个t有k组，每个y有2种状态，y1->y2,y2->y3,序列长3（k,i=1,2,y_i->y_i+1)

        # self.t=torch.tensor([[[[0,1],[0,0]],[[0,1],[0,0]]],
        #                      [[[1,0],[0,0]],[[0,0],[0,0]]],
        #                      [[[0,0],[0,0]],[[0,0],[1,0]]],
        #                      [[[0,0],[1,0]],[[0,0],[0,0]]],
        #                      [[[0,0],[0,0]],[[0,0],[0,1]]]],dtype=float);
        # self.lamb=torch.rand(self.k)
        # # 发射
        # # 序列长3，每个y和x出现的情况(l,t(y),状态）
        # self.s=torch.tensor([[[1,0],[0,0],[0,0]],
        #                      [[0,1],[0,1],[0,0]],
        #                      [[0,0],[1,0],[1,0]],
        #                      [[0,0],[0,0],[0,1]]],dtype=float)
        # self.mu = torch.rand(self.l)
        # 比上面多了个开始y0和结束y4
        # self.f=self.getf();

        # print(self.w)
        # f=self.f
        # self.P_y_x_condition_with_f([0,1,1])
        # self.f=f = torch.tensor([[[[0, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 0]]],
        #                        [[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        #                        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 0]]],
        #                        [[[0, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        #                        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 0]]],
        #                        [[[1, 0], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        #                        [[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        #                        [[[0, 0], [0, 0]], [[1, 0], [1, 0]], [[1, 0], [1, 0]], [[0, 0], [0, 0]]],
        #                        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 1], [0, 1]], [[0, 0], [0, 0]]]], dtype=float)
        # self.P_y_x_condition_with_f([0, 1, 1])

        # self.M = self.f2M();
        # print(self.M)
        # self.Z = self.Z_M()

    def get_ts(self, x_vec):
        """
                给定一个输入x_vec，计算这个输入上的所有的(y',y)组合的特征值。
                size: len(x_vec) + 1, Y, Y, K
                Axes:
                0 - T or time or sequence index
                1 - y' or previous label
                2 - y  or current  label
                3 - f(y', y, x_vec, i) for i s
                """
        self.f = torch.zeros(self.k,len(x_vec) + 1, self.y_size, self.y_size,dtype=float)
        for l, f in enumerate(self.func):
            for i in range(len(x_vec) + 1):
                for yp in range(self.y_size):
                    for y in range(self.y_size):
                        # print(i,j,k,l)
                        # print(yp, y, i,x_vec)
                        self.f[l,i, yp, y] = f(yp, y, x_vec, i)
        # print(result)

        return self.f





    def t_s2f(self):
        '''
        将self.t,self.s转化为简化形势下的参数f
        :return: f
        '''
        f = torch.cat((torch.cat((torch.zeros(self.k, 1, self.y_size, self.y_size, dtype=float), self.t), axis=1),
                       torch.zeros(self.k, 1, self.y_size, self.y_size, dtype=float)), axis=1)
        s = torch.zeros(self.l, self.self.sequence_len + 1, self.y_size, self.y_size, dtype=float)
        for i in range(self.l):
            for j in range(self.self.sequence_len):
                s[i, j, 0] = s[i, j, 1] = self.s[i, j]
        f = torch.cat((f, s), axis=0)
        return f

    def f2M(self):
        '''
        将简化形式下的参数f转化为矩阵形式
        :return: M
        '''
        M = torch.zeros(self.f.shape,dtype=float)
        # print(M[0])
        for i in range(self.k):
            M[i] = self.w[i] * self.f[i]
        # print(torch.sum(M, axis=0))
        M = torch.exp(torch.sum(M, axis=0))
        # f_ = torch.zeros(self.sequence_len + 1, self.y_size, self.y_size,dtype=float)
        # for i in range(self.k + self.l):
        #     f_ = f_+self.w[i] * self.f[i]
        # print(f_)
        # print("M(i,y_i-1,y_i):\n", M)
        # 因为y0=0,yn+1=0
        # 所以可以令M[0,1,0],M[0,1,1],M[3,0,1],M[3,1,1]=0
        M[0, 1:self.y_size] = torch.zeros(self.y_size - 1, self.y_size)
        M[self.sequence_len, :, 1:self.y_size] = torch.zeros(self.y_size, self.y_size - 1)
        # M[0, 1, 0] = M[0, 1, 1] = M[3, 0, 1] = M[3, 1, 1] = 0
        # print("M(i,y_i-1,y_i):\n", M)  # 与图对应上了--为了避免取log出现inf，就不对应了
        return M

    # 矩阵形式下求p(y|x)
    def P_y_x_condition_with_M(self, y):
        # y0=yn+1=0
        y.append(0)
        #
        y.insert(0, 0)
        # print(y)
        p = 1;
        for i in range(self.sequence_len + 1):
            # print(y[i],y[i+1],self.M[i, y[i], y[i + 1]])
            p *= self.M[i, y[i], y[i + 1]]
        # print(p)
        return p / self.Z

    # 简化参数形式下求p(y|x)
    def P_y_x_condition_with_f(self, y):
        # y0=yn+1=0
        y.append(0)
        # print(y)
        y.insert(0, 0)
        sum = 0
        for i in range(self.k):
            for j in range(len(y) - 1):
                sum += self.w[i] * self.f[i, j, y[j], y[j + 1]]
        return torch.exp(sum) / self.Z

    # 矩阵形式的归一化因子Z
    def Z_M(self):
        # print(self.M[0])
        z = self.M[0]
        for i in range(1, self.sequence_len + 1):
            z = torch.matmul(z, self.M[i])
        return z[0, 0]

    def Z_alpha(self, alpha):
        return torch.sum(alpha[self.sequence_len + 1])

    def Z_beta(self, beta):
        # print(beta)
        return torch.sum(beta[0])

    # 解码
    def Viterbi_M(self):
        delta = torch.zeros(self.sequence_len, self.y_size, dtype=float)
        logM = torch.log(self.M)
        # print(logM[0,0].shape)
        # print(delta.shape)
        # print(delta[0])
        delta[0] = logM[0, 0]
        # torch.max(delta[0].reshape(self.y_size, 1) + logM[1], axis=0)
        indices = []
        for i in range(1, self.sequence_len):
            # print(delta[i - 1].reshape(self.y_size, 1) + logM[i])
            delta[i], indice = torch.max(delta[i - 1].reshape(self.y_size, 1) + logM[i], axis=0)
            indices.append(indice)
        # print(delta)
        #     print(indices)
        path = torch.zeros(self.sequence_len, dtype=torch.long)
        #     print(path)
        path[self.sequence_len - 1] = torch.argmax(delta[self.sequence_len - 1])
        #     print(path)
        for i in range(self.sequence_len - 2, -1, -1):
            path[i] = indices[i][path[i + 1]]
        #     print(path)
        return path

    # 前向算法
    def alpha(self):
        alpha = torch.zeros(self.sequence_len + 2, self.y_size, dtype=float)
        alpha[0, 0] = 1
        for i in range(self.sequence_len + 1):
            alpha[i + 1] = torch.matmul(alpha[i].reshape(1, self.y_size), self.M[i])
        # print(alpha)
        return alpha

    def beta(self):
        beta = torch.zeros(self.sequence_len + 2, self.y_size, dtype=float)
        beta[self.sequence_len + 1, 0] = 1
        for i in range(self.sequence_len, -1, -1):
            #         print(M[i],beta[i+1].reshape(y_size,1))
            beta[i] = torch.matmul(self.M[i], beta[i + 1].reshape(self.y_size, 1)).reshape(self.y_size)
        # print(beta)
        return beta

    def p_y_x_condition_alpha_beta(self, alpha, beta):
        # p(y_i|x)
        p_y_x = alpha * beta / self.Z_alpha(alpha)
        #     print(alpha[2].reshape(1,y_size)*beta[2].reshape(y_size,1))
        return p_y_x

    def p_y12_x_condition_alpha_beta(self, alpha, beta):
        # p(y_{i-1}，y_i|x）
        p = self.M.clone().detach()
        for i in range(self.sequence_len + 1):
            p[i] = alpha[i].reshape(self.y_size, 1) * p[i] * beta[i + 1]
        return p / self.Z_alpha(alpha);

    def E_fk_py_x(self, k, alpha, beta):  # E_{p(y|x)}(f_k)
        return torch.sum(self.f[k] * self.p_y12_x_condition_alpha_beta(alpha, beta))


    def P_x_experience(self, x):
        p = torch.zeros(self.x_size)
        for row in x:
            for word in row:
                p[word] += 1
        p = p / torch.sum(p)
        # print(p)
        return p

    def delta_log_L(self,alpha,beta,y):
        # print(self.f[:,3,[0,0,1,1],[0,1,1,0]])
        #y=[0,1,1]
        delta=torch.sum(self.f[:,len(y),[0]+y,y+[9]],axis=(1))-torch.sum(self.f* self.p_y12_x_condition_alpha_beta(alpha, beta),axis=(1,2,3))
        return delta


    def train(self,traindata):
        delta=0
        batch_size=100
        num_batch=int(len(traindata[0])/batch_size)
        for e in range(num_batch):
            delta=0
            for i in range(batch_size):
                x = traindata[0][e*batch_size+i]
                y = traindata[1][e*batch_size+i]
                # print(x)
                self.sequence_len = len(x)
                self.get_ts(x)
                self.M=self.f2M()
                alpha = self.alpha()
                beta = self.beta()
                delta += self.delta_log_L(alpha, beta, y)
                # print(y)
                # print(self.Viterbi_M())
                predict=self.Viterbi_M()
                self.measure(predict,torch.tensor(y))
                # print(torch.tensor(y)==self.Viterbi_M())
            print(delta)
            self.w = self.w + 0.0001 * delta
        #     with open('./train/f_'+str(i), 'w', encoding='utf-8') as f:
        #         f.write(str(self.f.detach().numpy().tolist()))
        #     self.M=self.f2M()
        #     with open('./train/M_'+str(i), 'w', encoding='utf-8') as f:
        #         f.write(str(self.M.detach().numpy().tolist()))
        # delta=0
        # for i in range(1):
        #     print(i)
        #     k=self.k
        #     with open('./train/f_' + str(i), 'r', encoding='utf-8') as f:
        #         self.f=torch.tensor(eval(f.read()))
        #     with open('./train/M_' + str(i), 'r', encoding='utf-8') as f:
        #         self.M = torch.tensor(eval(f.read()))
        #     print(self.f)
        #     y=traindata[1][i]

    def measure(self, predict, y):
        acc = (torch.sum(torch.eq(predict, y))).type(torch.FloatTensor) / float(len(y))
        TP = torch.zeros(self.y_size, dtype=float)
        FP = torch.zeros(self.y_size, dtype=float)
        FN = torch.zeros(self.y_size, dtype=float)
        for i in range(len(y)):
            if (y[i] == predict[i]):
                TP[y[i]] += 1
            else:
                FP[predict[i]] += 1
                FN[y[i]] += 1
        # micro:算总的
        # print(torch.sum(TP))
        micro_precision = torch.sum(TP) / (torch.sum(TP) + torch.sum(FP))
        micro_recall = torch.sum(TP) / (torch.sum(TP) + torch.sum(FN))
        micro_F1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        # macro ：算每一类的然后平均
        TP[TP == 0] = self.epsilon
        FP[FP == 0] = self.epsilon
        FN[FN == 0] = self.epsilon
        macro_precision = TP / (TP + FP)
        macro_recall = TP / (TP + FN)
        # print("TP:", TP)
        # print("FN:", FN)
        # print("FP:", FP)
        # print("P:", macro_precision)
        # print("R:", macro_recall)
        # print("F1:", macro_recall * macro_precision)
        macro_F1 = 2 * (macro_recall * macro_precision) / (macro_recall + macro_precision)
        # print(macro_F1)
        macro_F1 = torch.mean(macro_F1)
        print(acc,micro_F1,macro_F1)
        return acc, micro_F1, macro_F1




if __name__=='__main__':
    traindata, dic_word_list, dic_label_list, dic_word, dic_label = getAllTrain()


    wordset = getword_set(len(dic_label), traindata)
    labels = list(range(len(dic_label)))
    # print(labels)
    model1 = CRF(len(dic_word), len(dic_label), get_feature_functions(wordset, labels, traindata[0]))
    model1.train(traindata)


