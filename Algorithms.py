import random
import numpy as np
import math
from Environment import *
from scipy.optimize import minimize
from itertools import product
import copy
    
class MaxWeight:
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
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



    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S    
    
    def __init__(self,seed,x,N,K,L,T,theta_list):
        print('Maxweight')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[]
        self.Q=np.zeros(N)
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.M=self.construct_M()
        self.theta=theta_list
    def run(self,t,index,Q):   
        if t==1:
            self.S=copy.deepcopy(random.choice(self.M))
            self.Q=Q    
        else:
            self.Q=Q    
            R=0
            tmp_R=0

            for partition in self.M:
                for k in range(self.K):
                    if len(partition[k])>0:                    
                        tmp_R+=np.sum(np.exp(self.x[partition[k]]@self.theta[k])@self.Q[partition[k]])/(1+np.sum(np.exp(self.x[partition[k]]@self.theta[k])))
                if R<tmp_R:
                    R=tmp_R
                    self.S=copy.deepcopy(partition)
                tmp_R=0
        self.remove_elements(self.S, self.Q)

    def offer(self):
        return self.S   
    
    def name(self):
        return 'MaxWeight'    
    

class UCB_QMB:

    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
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

    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta)  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.d] = theta[0:self.d] / norm_theta
            

        return theta
    
    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
    #         obj=self.g@(theta-theta_prev)+(1/2)*(theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev)    
        if len(S)>0:
            iter = 0
            theta=copy.deepcopy(theta_prev)
            while True:
                theta_=copy.deepcopy(theta)
                grad=self.g+V@(theta-theta_prev)
                theta = theta - 0.01*grad
                iter += 1
                if np.linalg.norm(theta-theta_) < 1e-5 or iter > 20:
                    break
            theta=self.project_to_unit_ball(theta)
        else:
            theta=theta_prev

        return theta   
  

    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S    
            
    def __init__(self,seed,x,N,K,L,T):
        print('UCB_QMB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[]
        self.d=len(self.x[0])
        self.alpha=1
        self.kappa=0.25
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.lamb=1
        self.h=np.zeros((N,K))
        self.V=[]
        self.z=[]
        self.Ur=[]
        self.Q=np.zeros(N)
        self.theta=np.zeros((self.K,self.d))
   
    
    def run(self,t,index,Q):   
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            for k in range(self.K):
                self.V.append(self.lamb*np.identity(self.d))
        else:
            y=self.match_elements(self.S, index) 

            for k in range(self.K):
                for n in self.S[k]:
                    self.V[k]+=(self.kappa/2)*np.outer(self.x[n],self.x[n])
                theta_prev=self.theta[k].copy()
                y_k=y[k]
                S_k=self.S[k]
                self.theta[k]=self.fit(theta_prev,self.x,y_k,S_k,self.V[k])

            self.alpha=np.sqrt(1+(self.d/self.kappa)*np.log(1+(t*self.L)/(self.d*self.lamb)))

            for k in range(self.K):
                for n in range(self.N):
                    self.h[n,k]=self.x[n]@self.theta[k]+self.alpha*np.sqrt(self.x[n]@np.linalg.inv(self.V[k])@self.x[n])
            self.Q=Q
            R=0
            tmp_R=0
            for partition in self.M:
                for k in range(self.K):
                    if len(partition[k])>0:                    
                        tmp_R+=np.sum(np.exp(self.h[partition[k],k])@self.Q[partition[k]])/(1+np.sum(np.exp(self.h[partition[k],k])))
                if R<tmp_R:
                    R=tmp_R
                    self.S=copy.deepcopy(partition)
                tmp_R=0
        self.remove_elements(self.S, self.Q)

    def offer(self):
        return self.S   
    
    def name(self):
        return 'UCB_QMB'    
    

class TS_QMB:

    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
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

    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta)  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.d] = theta[0:self.d] / norm_theta
            

        return theta
    
    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
    #         obj=self.g@(theta-theta_prev)+(1/2)*(theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev)    
        if len(S)>0:
            iter = 0
            theta=copy.deepcopy(theta_prev)
            while True:
                theta_=copy.deepcopy(theta)
                grad=self.g+V@(theta-theta_prev)
                theta = theta - 0.01*grad
                iter += 1
                if np.linalg.norm(theta-theta_) < 1e-5 or iter > 20:
                    break
            theta=self.project_to_unit_ball(theta)
        else:
            theta=theta_prev

        return theta   
  


    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S    
        
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,x,N,K,L,T):
        print('TS_QMB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[]
        self.d=len(self.x[0])
        self.alpha=1
        self.kappa=0.25
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.lamb=1
        self.h=np.zeros((N,K))
        self.V=[]
        self.z=[]
        self.Ur=[]
        self.Q=np.zeros(N)
        self.theta=np.zeros((self.K,self.d))

    def run(self,t,index,Q):   
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            for k in range(self.K):
                self.V.append(self.lamb*np.identity(self.d))
        else:
            y=self.match_elements(self.S, index) 

            for k in range(self.K):
                for n in self.S[k]:
                    self.V[k]+=(self.kappa/2)*np.outer(self.x[n],self.x[n])
                theta_prev=self.theta[k].copy()
                y_k=y[k]
                S_k=self.S[k]

                self.theta[k]=self.fit(theta_prev,self.x,y_k,S_k,self.V[k])

            M=math.ceil(1-(np.log(self.K*self.L)/np.log(1-1/(4*np.log(math.e*math.pi)))))
            self.beta=np.sqrt(1+(self.d/self.kappa)*np.log(1+(t*self.L)/(self.d*self.lamb)))

            for k in range(self.K):
                mean=self.theta[k]
                cov=self.beta**2*np.linalg.inv(self.V[k])
                theta_sample=np.random.multivariate_normal(mean, cov, M)
                for n in range(self.N):
                    self.h[n,k]=max(self.x[n]@theta_sample.T)
            self.Q=Q    
            R=0
            tmp_R=0
            for partition in self.M:
                for k in range(self.K):
                    if len(partition[k])>0:                    
                        tmp_R+=np.sum(np.exp(self.h[partition[k],k])@self.Q[partition[k]])/(1+np.sum(np.exp(self.h[partition[k],k])))
                if R<tmp_R:
                    R=tmp_R
                    self.S=copy.deepcopy(partition)
                tmp_R=0
        self.remove_elements(self.S, self.Q)


    def offer(self):
        return self.S   
    
    def name(self):
        return 'TS_QMB'    


##################



class Q_UCB:
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
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



    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S    
        
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,x,N,K,L,T):
        print('Q-UCB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[[] for _ in range(K)]
        self.Q=np.zeros(N)
        self.kappa=0.1
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.phi=np.zeros((N,K))
        self.n=np.zeros((N,K))
        self.mean=np.zeros((N,K))
        self.mu=np.zeros((N,K))
        self.ucb=np.zeros((N,K))
    def run(self,t,index,Q):   
        if t==1:
            self.S=[[] for _ in range(self.K)]
            K_list=list(range(self.K))
            for n in range(self.N):
                k=random.choice(K_list)
                self.S[k].append(n)
                K_list.remove(k)
                if len(K_list)==0:
                    break
        else:
            y=self.match_elements(self.S, index) 
            for k in range(self.K):
                for i,n in enumerate(self.S[k]):
                    self.n[n,k]+=1
                    self.mean[n,k]=((self.n[n,k]-1)*self.mean[n,k]+y[k][i])/self.n[n,k]
                        
            E=bernoulli(min(1,3*self.K*(math.log(t)**2)/2))
            if E==1:
                self.S=[[] for _ in range(self.K)]
                K_list=list(range(self.K))
                for n in range(self.N):
                    k=random.choice(K_list)
                    self.S[k].append(n)
                    K_list.remove(k)
            else:
                for k in range(self.K):
                    for n in range(self.N):
                        self.ucb[n,k]=self.mean[n,k]+np.sqrt(math.log(t)**2/(2*self.n[n,k]))
                assignment=[]
                for n in range(self.N):
                    assignment.append(np.argmax(self.ucb[n,:]))
                arm_remain=list(set(list(range(self.K)))-set(assignment))
                K_list=list(range(self.K))
                random.shuffle(K_list) 
                for k in K_list:  ##projection
                    while assignment.count(k)>1: #duplicated server selection
                        candidate=np.where(np.array(assignment)==k)[0].tolist()
                        if len(arm_remain)>0: #there is available server
                            j=random.choice(candidate) 
                            a=random.choice(arm_remain)
                            assignment[j]=a #re-assignemnt
                            arm_remain.remove(a) 
                        else:
                            j=random.choice(candidate) #selecting agent to be remained
                            candidate.remove(j) 
                            for index in candidate:
                                assignment[index]=self.K #assignemt null servers

                self.S=[[] for _ in range(self.K)]
                for n,k in enumerate(assignment):
                    if k!=self.K:
                        self.S[k].append(n)   #assorments based on the projection result

        self.Q=Q
        self.remove_elements(self.S, self.Q)  #remove agents having empty queues

    def offer(self):
        return self.S   
    
    def name(self):
        return 'Q-UCB'    



class DAM_UCB:

    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
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

    def loss(self, theta, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        g=0
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            g += p
        obj=g@(theta-theta_prev)+(1/2)*np.sqrt((theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev))    
        return obj 

    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        if len(S)>0:
            obj_func = lambda theta: self.loss(theta,theta_prev, *args)
            constraint = ({'type': 'ineq', 'fun': lambda theta: 1 - np.linalg.norm(theta, ord=2)})
            result = minimize(obj_func, theta_prev, constraints=constraint, method='SLSQP')
            theta_update=result.x
        else:
            theta_update=theta_prev

        return theta_update


    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S        
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
            
    def __init__(self,seed,x,N,K,L,T,eps):
        print('DAM_UCB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[[] for _ in range(K)]
        self.d=len(self.x[0])
        self.alpha=1
        self.kappa=0.1
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.lamb=1
        self.h=np.zeros((N,K))
        self.V=[]
        self.Q=np.zeros(N)
        self.theta=np.zeros((self.K,self.d))
        self.delta=1/(L+1)
        self.xi=eps**2/((K**2)*(np.log(N)+K))
        self.L_check=math.ceil(max(1,(2/np.log(1-self.delta))**2,np.log(self.xi)/np.log(1-self.delta)))
        print(self.L_check)
        self.L_conv=(1/100)*math.ceil((K*self.L_check/eps)*(np.log(N)+K)) # Constant parameter is tuned to improve performance.
        print(self.L_conv)
        self.L_ep=math.ceil(((1/eps)+1)*self.L_conv)
        print(self.L_ep)
        self.l=1
        self.mu=np.zeros((N,K))
        self.n=np.zeros((N,K))
        self.mu_tmp=np.zeros((N,K))
        self.n_tmp=np.zeros((N,K))
        self.ucb=np.zeros((N,K))
        self.start=True
        self.t_0=0
        self.tau=np.zeros(N)
        self.w=np.zeros((N,K))
        self.p=np.zeros((N,K))
        self.eta=np.zeros(N)
        self.epsilon=eps
        self.tau_update=[False]*N
        self.assign=np.zeros(self.N)
        print(self.L_ep)
    def run(self,t,index,Q):
        if t==1: 
            self.eta=np.random.uniform(0,10**(-9),self.N)
            self.M= self.construct_M()
            self.S_alg = copy.deepcopy(random.choice(self.M))
        if t>1: 
            y=self.match_elements(self.S, index) 
            for k in range(self.K):
                for i,n in enumerate(self.S[k]):
                    self.n_tmp[n,k]+=1
                    self.mu_tmp[n,k]=((self.n_tmp[n,k]-1)*self.mu_tmp[n,k]+y[k][i])/self.n_tmp[n,k] 
        if self.start==True:
            self.t_0=(self.l-1)*self.L_ep+1
            self.ucb=np.maximum(self.delta,np.minimum(1,self.mu+np.sqrt(3*np.log(self.t_0+self.K)/self.n)))
            self.start=False
            for n in range(self.N):
                self.tau[n]=self.t_0-1
            for k in range(self.K):
                self.w[:,k]=self.ucb[:,k]*Q
            self.p=np.zeros((self.N,self.K))
            
        if t<=self.t_0+self.L_conv-1:

            for n in range(self.N):
                if self.tau_update[n]==True or n in index:
                    self.tau[n]=t-1
                    self.tau_update[n]=False
            for n in range(self.N):

                if t-self.tau[n]>self.L_check:
                    self.tau_update[n]=True
                    j=np.argmax(self.w[n]-self.p[n])
                    if self.w[n,j]-self.p[n,j]>0:
                        self.p[n,j]=self.p[n,j]+(1/16)*self.epsilon*(1-self.eta[n])*self.w[n,j]
                        self.assign[n]=j
                        for i, inner_array in enumerate(self.S_alg):
                            if n in inner_array:
                                self.S_alg[i].remove(n)
                        self.S_alg[j].append(n)

                    else:
                        self.assign[n]=None
                        for i, inner_array in enumerate(self.S_alg):
                            if n in inner_array:
                                self.S_alg[i].remove(n)


        elif t>self.t_0+self.L_ep-1:
            self.l=self.l+1
            self.mu=self.mu_tmp.copy()
            self.n=self.n_tmp.copy()
            self.start=True

        self.Q=Q
        self.S=copy.deepcopy(self.S_alg)
        self.remove_elements(self.S, self.Q)
    def offer(self):
        return self.S   
    
    def name(self):
        return 'DAM-UCB'    
    


class MaxWeight_UCB:
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
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



    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S    
        
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,x,N,K,L,T):
        print('Maxweight-UCB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[[] for _ in range(K)]
        self.Q=np.zeros(N)
        self.kappa=0.1
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.phi=np.zeros((N,K))
        self.n=np.zeros((N,K))
        self.m=np.zeros((N,K))
        self.mu=np.zeros((N,K))
        self.b=np.zeros((N,K))
    def run(self,t,index,Q):   
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            self.Q=Q    
        else:
            y=self.match_elements(self.S, index) 
            for k in range(self.K):
                for i,n in enumerate(self.S[k]):
                    self.m[n,k]+=1
                    if y[k][i]==1:
                        self.n[n,k]=self.n[n,k]+y[k][i]
                        self.phi[n,k]=self.phi[n,k]+self.m[n,k]
                        self.m[n,k]=0
            self.Q=Q
            self.mu=self.n/self.phi
            self.b=1/self.kappa*np.sqrt(np.log(t-1)/self.n)
            self.S=[[] for _ in range(self.K)]
            k_set=[[] for _ in range(self.N)]
            for k in range(self.K):
                j=np.argmax(Q/np.maximum(1/self.mu[:,k]-self.b[:,k],1))
                k_set[j].append(k)
            for j in range(self.N):
                if len(k_set[j])!=0:
                    k_sample=random.sample(k_set[j],k=1)[0]
                    self.S[k_sample]=[j]

                    
        self.remove_elements(self.S, self.Q)


    def offer(self):
        return self.S   
    
    def name(self):
        return 'MaxWeight-UCB'    
    



class ETC_GS:
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
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
                
    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
 
    def generate_preferences(self, ucb):
        n, k = ucb.shape
        men_preferences = [list(np.argsort(-ucb[n_, :])) for n_ in range(n)]  # Sorting in descending order
        women_preferences = [list(np.argsort(-ucb[:, k_])) for k_ in range(k)]  # Sorting in descending order
        return men_preferences, women_preferences


    def stable_matching(self, men_preferences, women_preferences):
        # Number of men and women
        n = len(men_preferences)
        k = len(women_preferences)
        # Initialize arrays to store matching
        engaged_to = [None] * n  # engaged_to[w] represents the man engaged to woman w
        men_status = [0] * n  # men_status[m] represents the index of the next woman to propose to
        pre_engaged= [None] * n
        while None in engaged_to:            
            for man in range(n):
                if engaged_to[man] is None:
                    # Get the next woman to propose to
                    woman = men_preferences[man][men_status[man]]
                    men_status[man] += 1

                    # Check if the woman is not engaged
                    if woman not in engaged_to:
                        engaged_to[man] = woman
                    else:
                        # Woman is engaged, check her preferences
                        current_man = engaged_to.index(woman)
                        if women_preferences[woman].index(man) < women_preferences[woman].index(current_man):
                            # Woman prefers the current proposal
                            engaged_to[current_man] = None
                            engaged_to[man] = woman
            if pre_engaged==engaged_to:
                break
            pre_engaged=engaged_to
        return engaged_to
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,x,N,K,L,T,delta):
        print('ETC_GS')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[]
        self.r=0
        self.alpha=1
        self.kappa=0.2
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.lamb=1
        self.h=np.zeros((N,K))
        self.V=[]
        self.mean=np.zeros((self.N,self.K))
        self.n=np.ones((self.N,self.K))
        self.ucb=np.zeros((self.N,self.K))
        self.explore=True
        self.delta=delta
        self.h=4/delta**2*math.log(1+(T*delta**2*N/4))
        self.Q=np.zeros(N)

    def run(self,t,index,Q):   
        if self.explore==True:
            if t==1:   
                self.M=self.construct_M()
            if t!=1:
                y=self.match_elements(self.S, index) 
                for k in range(self.K):
                    for i,n in enumerate(self.S[k]):
                        self.n[n,k]+=1
                        self.mean[n,k]=((self.n[n,k]-1)*self.mean[n,k]+y[k][i])/self.n[n,k]      
            self.S=[[] for _ in range(self.K)]
            self.S=copy.deepcopy(random.choice(self.M))
            if t>self.h*self.K:
                men_preferences, women_preferences = self.generate_preferences(self.mean)
                result = self.stable_matching(men_preferences, women_preferences)
                self.S=[[] for _ in range(self.K)]
                for n,k in enumerate(result):
                    if k!= None:
                        self.S[k].append(n)
                self.explore=False

        self.remove_elements(self.S, self.Q)
    def offer(self):
        return self.S   
    
    def name(self):
        return 'ETC-GS'    
    
    