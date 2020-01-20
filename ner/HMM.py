import torch
import torch.nn as nn
import torch.optim as optim

from processData import *
class HMM(nn.Module):
    def __init__(self,x_size,y_size):
        self.y_size=y_size;
        self.x_size=x_size;
        self.transition=torch.randn(y_size,y_size);#转移概率p(y|y)
        self.b=torch.randn(y_size,x_size);#发射概率P（x|y)
        self.pi=torch.randn(y_size,1);#先验
        self.transition=torch.tensor([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
        self.b=torch.tensor([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
        self.pi=torch.tensor([[0.2],[0.4],[0.4]])


    def alpha(self,x):#前向算法p(x1,x2,x3,...,xt,yt)
        alpha=(self.b[:,x[0]]*self.pi[:].reshape(self.y_size)).reshape(1,self.y_size)
        # print(alpha)
        for i in range(1,len(x)):
            alpha=torch.cat((alpha,(torch.matmul(alpha[i-1],self.transition)*self.b[:,x[i]]).reshape(1,self.y_size)),0)
        return alpha

    def beta(self,x):
        beta=torch.ones(1,self.y_size)
        for i in range(len(x)-2,-1,-1):
            beta=torch.cat((torch.sum(beta[0]*self.transition*self.b[:,x[i+1]],axis=1).reshape(1,self.y_size),beta))
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
        V=self.b[:,x[0]]*self.pi[:].reshape(self.y_size);
        list=[]
        # print(V)
        # 前向计算各部分概率
        for t in range(1,len(x)):
            # max,indices=torch.max(V[t - 1].reshape(self.y_size, 1) * self.transition, axis=0)
            # list.append(indices)
            # V=torch.cat((V,(self.b[:,x[t]]*max).reshape(1,self.y_size)),axis=0)
            max, indices = torch.max(V.reshape(self.y_size,1) * self.transition, axis=0)
            list.append(indices)
            V=self.b[:,x[t]]*max
            # print(V)
        #后向寻找路径
        # print(list)
        max,indices=torch.max(V,axis=0)
        path=indices.reshape(1)
        for i in range(len(list)-1,-1,-1):
            path=torch.cat((list[0][path[0]].reshape(1),path))
        return path;#y1=path0,y2=path1

    def gamma(self,alpha,beta,p_x):
        # 返回：gamma_ij,p(yj|xi),行为x,列为y
        return alpha*beta/p_x

    def xi(self,x,alpha,beta):
        # print(alpha_yt,self.b[y_t1,x_t1],beta_yt1,self.transition[y_t,y_t1])
        # return alpha_yt*self.b[y_t1,x_t1]*beta_yt1*self.transition[y_t,y_t1]/p_x
        # return (t(xt),yt,yt+1)
        xi=[]
        for t in range(0,len(x)-1):
            xi.append((alpha[t].reshape(self.y_size,1)*self.transition*self.b[:,x[t+1]]*beta[t+1]))
            # print(xi[t])
        return torch.cat(xi).reshape(len(xi),self.y_size,self.y_size)

    def EM(self,traindata):#train
        for x in traindata:
            #E
            alpha = self.alpha(x)
            p_x=self.p(x,alpha)
            beta=self.beta(x)
            gamma=self.gamma(alpha,beta,p_x)
            xi=self.xi(x,alpha,beta)
            # print(gamma)
            #M
            #对时间求和了从t=1到t
            gamma_sum=torch.sum(gamma,axis=0).reshape(self.y_size,1)
            # print(gamma_sum)
            self.b=torch.zeros(self.y_size,self.x_size)
            for t in range(0,len(x)):
                # print(self.b[:,x[t]])
                self.b[:,x[t]]+=gamma[t]
            # print(self.b)
            self.b=self.b/gamma_sum
            # 从t=1加到t-1
            gamma_sum_=gamma_sum-gamma[len(x)-1].reshape(self.y_size,1)
            # print(gamma_sum_)
            # print(torch.sum(xi,axis=0))
            self.transition=torch.sum(xi, axis=0)/gamma_sum_












# train=readFile("./CoNLL-2003/eng.train");
# print(train)


# print(readFile("./CoNLL-2003/eng.train");)

model=HMM(2,3)
x=[0,1,0]
# alpha=model.alpha(x);
# beta=model.beta(x);
# p_x = model.p(x, alpha);
# gamma=model.gamma(alpha,beta,p_x)
# xi=model.xi(x,alpha,beta);
# Viterbi=model.Viterbi(x)
# print(alpha)
# print(beta)
# print(gamma)
traindata=[x]
model.EM(traindata)


