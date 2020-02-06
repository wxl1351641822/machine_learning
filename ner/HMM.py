import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from processData import *
class HMM(nn.Module):
    def __init__(self,x_size,y_size):# y-label,x-word
        self.y_size=y_size;
        self.x_size=x_size;
        # print(self.x_size, self.y_size)
        self.transition=torch.rand(y_size,y_size);#转移概率p(y|y)
        self.transition=self.transition/torch.sum(self.transition,axis=1).reshape(self.y_size,1);
        self.b=torch.rand(y_size,x_size);#发射概率P（x|y)
        self.b=self.b/torch.sum(self.b, axis=1).reshape(self.y_size,1);
        print(self.b.shape)
        # print(torch.sum(self.b, axis=1))
        self.pi=torch.rand(y_size,1);#先
        self.pi=self.pi/torch.sum(self.pi)
        self.epsilon=1e-8
        # print(torch.sum(self.pi))
        # self.transition=torch.tensor([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
        # self.b=torch.tensor([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
        # self.pi=torch.tensor([[0.2],[0.4],[0.4]])


    def alpha(self,x):#前向算法p(x1,x2,x3,...,xt,yt)
        alpha=(self.b[:,x[0]]*self.pi[:].reshape(self.y_size)).reshape(1,self.y_size)
        # print(alpha)
        for i in range(1,len(x)):
            alpha=torch.cat((alpha,(torch.matmul(alpha[i-1],self.transition)*self.b[:,x[i]]).reshape(1,self.y_size)),0)
            # print(alpha[i-1])
            # print(self.transition)
        return alpha

    def beta(self,x):
        beta = torch.ones(1, self.y_size)
        for i in range(len(x) - 2, -1, -1):
            beta = torch.cat((torch.sum(beta[0] * self.transition * self.b[:, x[i + 1]], axis=1).reshape(1, self.y_size), beta))
        return beta
    def alpha_scaling(self,x):#前向算法p(x1,x2,x3,...,xt,yt)
        # print(self.b[:])
        alpha=(self.b[:,x[0]]*self.pi[:].reshape(self.y_size)).reshape(1,self.y_size)

        # print(alpha)
        for i in range(1,len(x)):
            alpha[i-1]=alpha[i-1] / torch.sum(alpha[i-1], axis=0)
            alpha=torch.cat((alpha,(torch.matmul(alpha[i-1],self.transition)*self.b[:,x[i]]).reshape(1,self.y_size)),0)
            # print(alpha[i-1],torch.matmul(alpha[i-1],self.transition))
            # print()
            # print(self.transition)
        alpha[len(alpha)-1]=alpha[len(alpha)-1] / torch.sum(alpha[len(alpha)-1], axis=0)
        return alpha

    def beta_scaling(self,x):
        beta = torch.tensor([[1/self.y_size]*self.y_size])
        for i in range(len(x) - 2, -1, -1):
            k=torch.sum(beta[0] * self.transition * self.b[:, x[i + 1]], axis=1)
            beta = torch.cat(((k/torch.sum(k)).reshape(1, self.y_size), beta))
        return beta

    def p(self,x,alpha):#p(x)
        # alpha=self.alpha(x);
        return torch.sum(alpha[len(x)-1])

    # def gamma(self,alpha_yt,beta_yt,p_x):        # print(alpha)
    #     # print(beta)
    #     gamma=alpha_yt*beta_yt/p_x#alpha[t,y]*beta[t,y]
    #     # print(gamma)
    #     return gamma

    def Viterbi(self,x):#贪婪
        # print(x[0])
        # print(self.b[:,x[0]])
        # print(self.pi[:])
        b=torch.log(self.b)
        pi=torch.log(self.pi)
        transition=torch.log(self.transition)
        # print(x,b.shape)
        # print(b[:,x[0]])
        V=b[:,x[0]]+pi[:].reshape(self.y_size);
        list=[]
        # print(V)
        # 前向计算各部分概率
        for t in range(1,len(x)):
            # max,indices=torch.max(V[t - 1].reshape(self.y_size, 1) * self.transition, axis=0)
            # list.append(indices)
            # V=torch.cat((V,(self.b[:,x[t]]*max).reshape(1,self.y_size)),axis=0)
            # print(V.reshape(self.y_size,1))
            # print(self.transition)
            # print(V.reshape(self.y_size,1) +self.transition)
            max, indices = torch.max(V.reshape(self.y_size,1) +transition, axis=0)
            # print(max)
            list.append(indices)
            V=b[:,x[t]]+max
            # print(V)
        #后向寻找路径
        # print(list)
        max,indices=torch.max(V,axis=0)
        path=indices.reshape(1)
        for i in range(len(list)-1,-1,-1):
            path=torch.cat((list[0][path[0]].reshape(1),path))
        # print(path)
        return path;#y1=path0,y2=path1

    def gamma(self,alpha,beta,p_x):
        # 返回：gamma_ij,p(yj|xi),行为x,列为y
        return (alpha*beta)/p_x

    def xi(self,x,alpha,beta,p_x):
        # print(alpha_yt,self.b[y_t1,x_t1],beta_yt1,self.transition[y_t,y_t1])
        # return alpha_yt*self.b[y_t1,x_t1]*beta_yt1*self.transition[y_t,y_t1]/p_x
        # return (t(xt),yt,yt+1)
        xi=[]
        for t in range(0,len(x)-1):
            xi.append((alpha[t].reshape(self.y_size,1)*self.transition*self.b[:,x[t+1]]*beta[t+1])/p_x)
            # print(xi[t])
        return torch.cat(xi).reshape(len(xi),self.y_size,self.y_size)

    def EM(self,x):#x是一个批次，可以是多个句子
        #E
        alpha = self.alpha_scaling(x)
        # print(alpha)
        p_x=self.p(x,alpha)
        # print(p_x)
        beta=self.beta_scaling(x)
        # print(beta)
        gamma=self.gamma(alpha,beta,p_x)
        # print(gamma)
        xi=self.xi(x,alpha,beta,p_x)
        # print(xi)
        # print(gamma)
        #M
        #对时间求和了从t=1到t
        gamma_sum=torch.sum(gamma,axis=0).reshape(self.y_size,1)
        # print(gamma_sum)
        self.b=torch.ones(self.y_size,self.x_size)
        for t in range(0,len(x)):
            # print(self.b[:,x[t]])
            self.b[:,x[t]]+=gamma[t]
        self.b=self.b/torch.sum(self.b,axis=1).reshape(self.y_size,1)
        # 从t=1加到t-1
        gamma_sum_=gamma_sum-gamma[len(x)-1].reshape(self.y_size,1)
        # print(gamma_sum_)
        # print(torch.sum(xi,axis=0))
        self.transition=torch.sum(xi, axis=0)/torch.sum(torch.sum(xi, axis=0),axis=1).reshape(self.y_size,1)
        # self.transition=self.transition/torch.sum(self.transition,axis=1).reshape(self.y_size,1)

        # print(self.b)

    def getTransitionFromData(self,x):
        epsilon = 1e-8
        self.transition = torch.zeros(self.y_size, self.y_size)
        self.pi=torch.zeros(self.y_size,1)
        for row in x:
            # print(x[i,1],,x[i+1,1])
            self.pi[row[0]]+=1;
            for i in range(len(row)-1):

                self.transition[row[i], row[i + 1]] += 1
        self.transition[self.transition==0]=self.epsilon
        self.pi[self.pi==0]==self.epsilon
        self.pi=self.pi/torch.sum(self.pi)
        # print(self.transition)
        # print(torch.sum(transition,axis=1))
        # print()
        # print(1/45,15/45)
        self.transition=self.transition / torch.sum(self.transition, axis=1).reshape(self.y_size, 1)
        return self.transition,self.pi
        # torch.sum(transition[0])

    def getb(self,x):

        self.b = torch.zeros(self.y_size,self.x_size)
        for row in range(len(x[0])):
            for i in range(len(x[0][row])):
                self.b[x[1][row][i], x[0][row][i]] += 1
        self.b[self.b==0]=self.epsilon
        # print(self.b)
        # print(torch.sum(b,axis=1))
        # print(b/torch.sum(b,axis=1).reshape(len_label,1))
        self.b=self.b /torch.sum(self.b, axis=1).reshape(self.y_size, 1)
        return self.b

    def trainbyP(self,x,train_size):#u
        self.getTransitionFromData(x[1][0:train_size])
        self.getb([x[0][0:train_size],x[1][0:train_size]])
        # print(self.transition)
        # print(x[0][train_size:len(x[0])])
        # print(self.transition)#2.3000e+01, 1.0000e+00, 5.7020e+03, 7.0000e+00, 1.0000e+00, 1.2000e+01,         2.0000e+00, 9.1500e+02]])
        # print(self.b)

        predict=self.Viterbi(x[0][train_size]).reshape(1,len(x[0][train_size]))
        y=torch.tensor([x[1][train_size]])
        for i in tqdm(range(train_size+1,len(x[0]))):
            predict=torch.cat((predict,self.Viterbi(x[0][i]).reshape(1,len(x[0][i]))),axis=1)
            # print(predict)
            y=torch.cat((y,torch.tensor([x[1][i]])),axis=1)
        # print(y)
        # print(predict)
        # print(predict)
        # print(y)
        # print(torch.sum(torch.eq(predict,y)))
        # print(len(y))
        print("acc,macro-F1,micro-F1:",self.measure(predict[0],y[0]))

    def trainbyEM(self,x,train_size):
        for epoch in range(1):
            print("epoch",epoch,":")
            # print(x[0][0])
            word=[]
            for i in range(0,train_size):
                word+=(x[0][i])
            self.EM(word)
            predict = self.Viterbi(x[0][train_size]).reshape(1, len(x[0][train_size]))
            y = torch.tensor([x[1][train_size]])
            for i in tqdm(range(train_size,len(x[0]))):
                # self.EM(x[0][i])
                predict = torch.cat((predict, self.Viterbi(x[0][i]).reshape(1, len(x[0][i]))), axis=1)
                # print(predict)
                y = torch.cat((y, torch.tensor([x[1][i]])), axis=1)
                # print(y)
                if(i%100==0):
                    print("acc,macro-F1,micro-F1:",self.measure(predict[0],y[0]))

    def measure(self,predict,y):
        acc = (torch.sum(torch.eq(predict, y))).type(torch.FloatTensor) / float(len(y))
        TP=torch.zeros(self.y_size,dtype=float)
        FP=torch.zeros(self.y_size,dtype=float)
        FN=torch.zeros(self.y_size,dtype=float)
        for i in range(len(y)):
            if(y[i]==predict[i]):
                TP[y[i]]+=1
            else:
                FP[predict[i]]+=1
                FN[y[i]]+=1
        # micro:算总的
        print(torch.sum(TP))
        micro_precision=torch.sum(TP)/(torch.sum(TP)+torch.sum(FP))
        micro_recall=torch.sum(TP)/(torch.sum(TP)+torch.sum(FN))
        micro_F1=2*(micro_precision*micro_recall)/(micro_precision+micro_recall)
        # macro ：算每一类的然后平均
        TP[TP==0]=self.epsilon
        FP[FP==0]=self.epsilon
        FN[FN==0]=self.epsilon
        macro_precision=TP/(TP+FP)
        macro_recall=TP/(TP+FN)
        print("TP:",TP)
        print("FN:",FN)
        print("FP:",FP)
        print("P:",macro_precision)
        print("R:",macro_recall)
        print("F1:",macro_recall*macro_precision)
        macro_F1=2*(macro_recall*macro_precision)/(macro_recall+macro_precision)
        # print(macro_F1)
        macro_F1=torch.mean(macro_F1)
        return acc,micro_F1,macro_F1







traindata,dic_word_list,dic_label_list,dic_word,dic_label=getAllTrain()
model=HMM(len(dic_word_list),len(dic_label_list))
train_size=int(len(traindata[0])/5*4)
print("频次统计计算:")
model.trainbyP(traindata,train_size)
print("EM计算：")
model1=HMM(len(dic_word_list),len(dic_label_list))
model1.trainbyEM(traindata,train_size)









# train=readFile("./CoNLL-2003/eng.train");
# print(train)


# print(readFile("./CoNLL-2003/eng.train");)
#
# model=HMM(2,3)
# x=[0,1,0]
# alpha=model.alpha(x);
# alpha=model.alpha_scaling(x);
# print(alpha)
# beta=model.beta_scaling(x);
# print(beta)
# beta=model.beta(x);
# p_x = model.p(x, alpha);
# gamma=model.gamma(alpha,beta,p_x)
# xi=model.xi(x,alpha,beta,p_x);
# Viterbi=model.Viterbi(x)
# print(Viterbi)
# print(alpha)
# print(beta)
# print(gamma)
# traindata=[x]
# model.EM(traindata)


