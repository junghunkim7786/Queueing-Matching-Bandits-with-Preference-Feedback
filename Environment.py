import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from scipy.stats import bernoulli
from itertools import combinations
from itertools import product

class linear_Env:
    def __init__(self,seed,d,N,K,L,eps):
        np.random.seed(seed)
        self.d=d
        self.N=N
        self.K=K
        self.L=L
        self.x=np.zeros((self.N,self.d)) ## observed feature
        self.theta=np.zeros((self.K,self.d))
        self.lamb=np.zeros(self.N)
        self.Q=np.zeros(self.N)
        self.epsilon=eps
        while 0 in self.lamb:
            for n in range(self.N):
                self.x[n]=np.random.uniform(-1,1,self.d)
                self.x[n]=self.x[n]/np.sqrt(np.sum(self.x[n]**2))
            for k in range(self.K):
                self.theta[k]=np.random.uniform(-1,1,self.d)
                self.theta[k]=self.theta[k]/np.sqrt(np.sum(self.theta[k]**2))
            
            self.w=self.x@self.theta.T
            self.index=np.zeros(self.K)
            self.p=np.zeros(self.K+1)
            S=self.oracle()[1]
            
            for k in range(self.K):
                if len(S[k])>0:
                    for n in S[k]:
                        self.lamb[n]=max((np.exp(self.w[n,k]))/(1+np.sum(np.exp(self.w[S[k],k])))-self.epsilon,0)
    def delta(self):
        delta=2
        delta_tmp=0
        for n in range(self.N):
            sort_list=np.sort(self.x[n]@self.theta.T)
            delta_tmp=sort_list[-1]-sort_list[-2]
            if delta_tmp<delta:
                delta=delta_tmp
        return delta         
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
            for k in range(self.K):
                A_[n].append(k)
                
        combinations=list(product(*A_))
        for combination in combinations:
            S=[[] for _ in range(self.K)]
            for n in range(len(combination)):
                if combination[n]!=None: 
                    S[combination[n]].append(n)
            if all(len(sublist) <= self.L for sublist in S) and sum(len(sublist) for sublist in S)>=min(self.L*self.K,self.N):
                M.append(S)
        return M

    
    def oracle(self):
        oracle_reward=0
        tmp_oracle_reward=0
        M=self.construct_M()
        for partition in M:
            for k in range(self.K):
                if len(partition[k])>0:
                    for n in partition[k]:
                        tmp_oracle_reward+=np.sum(np.exp(self.w[partition[k],k]))/(1+np.sum(np.exp(self.w[partition[k],k])))
            if oracle_reward<tmp_oracle_reward:
                oracle_reward=tmp_oracle_reward
                S=partition
            tmp_oracle_reward=0
        return oracle_reward, S

    def observe(self,S):
        index=[]
        for k in range(self.K):
            if len(S[k])==0:
                index.append(None)
            else:
                prob=np.zeros(len(S[k])+1)
                prob[1:1+len(S[k])]=np.exp(self.w[S[k],k])/(1+np.sum(np.exp(self.w[S[k],k])))
                prob[0]=1- np.sum(prob[1:1+len(S[k])])
                index_list=np.insert(S[k], 0, self.N)
                x=np.random.choice(index_list, p=prob)
                index.append(x)
        for n in range(self.N):
            self.Q[n]=self.Q[n]+bernoulli.rvs(self.lamb[n], size=1)
  

        for i in index:
            if  i!=self.N and i is not None:
                if self.Q[i]>0:
                    self.Q[i]=self.Q[i]-1
        return index,self.Q                 

    def exp_reward(self,S,Q):
        R=0
        for k in range(self.K):
            R+=np.sum(np.exp(self.x[S[k]]@self.theta[k])@Q[S[k]])/(1+np.sum(np.exp(self.x[S[k]]@self.theta[k])))
        return R

