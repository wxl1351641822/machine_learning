import torch
import torch.nn as nn
from processData import *
# https://www.cnblogs.com/pinard/p/7048333.html
class CRF(nn.Module):
    def __init__(self,y_size,sequence_len,k,l):
        self.sequence_len = sequence_len;
        self.y_size = y_size;
        self.k=k
        self.l=l
        #转移
        #每个t有k组，每个y有2种状态，y1->y2,y2->y3,序列长3（k,i=1,2,y_i->y_i+1)
        self.t=torch.tensor([[[[0,1],[0,0]],[[0,1],[0,0]]],
                             [[[1,0],[0,0]],[[0,0],[0,0]]],
                             [[[0,0],[0,0]],[[0,0],[1,0]]],
                             [[[0,0],[1,0]],[[0,0],[0,0]]],
                             [[[0,0],[0,0]],[[0,0],[0,1]]]],dtype=float);
        self.lamb=torch.tensor([1,0.5,1,1,0.2])
        # 发射
        # 序列长3，每个y和x出现的情况(l,t(y),状态）
        self.s=torch.tensor([[[1,0],[0,0],[0,0]],
                             [[0,1],[0,1],[0,0]],
                             [[0,0],[1,0],[1,0]],
                             [[0,0],[0,0],[0,1]]],dtype=float)
        self.mu = torch.tensor([1, 0.5, 0.8, 0.5])
        #比上面多了个开始y0和结束y4
        self.f=self.t_s2f();
        self.w = torch.tensor([1, 0.5, 1, 1, 0.2, 1, 0.5, 0.8, 0.5],dtype=float)
        f=self.f
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

        self.M = self.f2M();
        # print(self.M)
        self.Z = self.Z_M()


    def t_s2f(self):
        '''
        将self.t,self.s转化为简化形势下的参数f
        :return: f
        '''
        f=torch.cat((torch.cat((torch.zeros(self.k,1,self.y_size,self.y_size,dtype=float),self.t),axis=1),torch.zeros(self.k,1,self.y_size,self.y_size,dtype=float)),axis=1)
        s=torch.zeros(self.l,self.sequence_len+1,self.y_size,self.y_size,dtype=float)
        for i in range(self.l):
            for j in range(self.sequence_len):
                s[i,j,0]=s[i,j,1]=self.s[i,j]
        f=torch.cat((f,s),axis=0)
        return f

    def f2M(self):
        '''
        将简化形式下的参数f转化为矩阵形式
        :return: M
        '''
        M = self.t_s2f();
        # print(M[0])
        for i in range(self.k + self.l):
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
        M[0,1:self.y_size]=torch.zeros(self.y_size-1,self.y_size)
        M[self.sequence_len,:,1:self.y_size]=torch.zeros(self.y_size,self.y_size-1)
        # M[0, 1, 0] = M[0, 1, 1] = M[3, 0, 1] = M[3, 1, 1] = 0
        # print("M(i,y_i-1,y_i):\n", M)  # 与图对应上了--为了避免取log出现inf，就不对应了
        return M


    # 矩阵形式下求p(y|x)
    def P_y_x_condition_with_M(self,y):
        # y0=yn+1=0
        y.append(0)
        #
        y.insert(0, 0)
        # print(y)
        p = 1;
        for i in range(self.sequence_len+1):
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
        sum=0
        for i in range(self.k+self.l):
            for j in range(len(y)-1):
                sum+=self.w[i]*self.f[i,j,y[j],y[j+1]]
        return torch.exp(sum)/self.Z

    # 一般参数形势下求p(y|x)
    def P_y_x_condition(self,y):# 参数形式
        sumt=0
        sums=0
        for i in range(self.k):
            for j in range(len(y)-1):
                sumt+=self.lamb[i]*self.t[i,j,y[j],y[j+1]]
                # print(i,j,self.lamb[i]*self.t[i,j,y[j],y[j+1]])
        for i in range(self.l):
            for j in range(len(y)):
                sums+=self.mu[i]*self.s[i,j,y[j]]
        return torch.exp(sums+sumt)/self.Z
    # 矩阵形式的归一化因子Z
    def Z_M(self):
        # print(self.M[0])
        z = self.M[0]
        for i in range(1, self.sequence_len + 1):
            z = torch.matmul(z, self.M[i])
        return z[0, 0]

    def Z_alpha(self,alpha):
        return torch.sum(alpha[self.sequence_len + 1])

    def Z_beta(self,beta):
        # print(beta)
        return torch.sum(beta[0])


    # 解码
    def Viterbi_M(self):
        delta = torch.zeros(3, 2,dtype=float)
        logM = torch.log(self.M)
        delta[0] = logM[0, 0]
        # torch.max(delta[0].reshape(self.y_size, 1) + logM[1], axis=0)
        indices = []
        for i in range(1, self.sequence_len):
            # print(delta[i - 1].reshape(self.y_size, 1) + logM[i])
            delta[i], indice = torch.max(delta[i - 1].reshape(self.y_size, 1) + logM[i], axis=0)
            indices.append(indice)
        # print(delta)
        #     print(indices)
        path = torch.zeros(self.sequence_len, dtype=torch.int)
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

    def p_y_x_condition_alpha_beta(self,alpha, beta):
        # p(y_i|x)
        p_y_x = alpha * beta / self.Z_alpha(alpha)
        #     print(alpha[2].reshape(1,y_size)*beta[2].reshape(y_size,1))
        return p_y_x

    def p_y12_x_condition_alpha_beta(self,alpha, beta):
        #p(y_{i-1}，y_i|x）
        p = self.M.clone().detach()
        for i in range(self.sequence_len + 1):
            p[i] = alpha[i].reshape(self.y_size, 1) * p[i] * beta[i + 1]
        return p / self.Z_alpha(alpha);

    def E_fk_py_x(self,k, alpha, beta):  # E_{p(y|x)}(f_k)
        return torch.sum(self.f[k] * self.p_y12_x_condition_alpha_beta(alpha, beta))


model=CRF(2,3,5,4)
y=[0,1,1]

print("求p(y|x)")
p_y_x_con=model.P_y_x_condition(y)
print("一般形式p(y|x)=p(y1=1,y2=2,y3=3|x)=",p_y_x_con)
y=[0,1,1]
p_y_x_con=model.P_y_x_condition_with_f(y)
print("简化形式p(y|x)=p(y1=1,y2=2,y3=3|x)=",p_y_x_con)
y=[0,1,1]
p_y_x_con=model.P_y_x_condition_with_M(y)
print("矩阵形式p(y|x)=p(y1=1,y2=2,y3=3|x)=",p_y_x_con)



##维特比解码
print("维特比解码，获得最优路径:",model.Viterbi_M())

# 前向算法
alpha=model.alpha()
print("前向算法alpha:\n",alpha)
# 后向算法
beta=model.beta()
print("后向算法alpha:\n",beta)

print("由alpha得到的Z",model.Z_alpha(alpha))
print("由beta得到的Z",model.Z_beta(beta))

print("p(yi|x):\n",model. p_y_x_condition_alpha_beta(alpha, beta))
print("p(y_{i-1}，y_i|x):\n",model. p_y12_x_condition_alpha_beta(alpha, beta))
print("E_{p(y|x)}(f_k):\n",model.E_fk_py_x(1,alpha,beta))
