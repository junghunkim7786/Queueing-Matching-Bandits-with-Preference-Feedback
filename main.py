from Environment import *
from Algorithms import *
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import multiprocessing



def run(T,repeat,d,N,K,L,i,eps):
    Q_sum=dict()
    avg_Q_sum=dict()
    oracle_reward=0
    Q_sum_list=dict()
    std=dict()
    index=dict()
    Q=dict()
    Env_dict=dict()
    S=dict()
    exp_reward=dict()
    avg_regret_sum=dict()
    oracle_reward=dict()
    regret_sum_list=dict()
    std_R=dict()
    for algorithm in ['MaxWeight','UCB_QMB','TS_QMB','ETC-GS','DAM-UCB','MaxWeight-UCB','Q-UCB']:
        Q_sum_list[algorithm]=np.zeros((repeat,T),float)
        avg_Q_sum[algorithm]=np.zeros(T,float)
        std[algorithm]=np.zeros(T,float)

        exp_reward[algorithm]=[]
        oracle_reward[algorithm]=[] 
        regret_sum_list[algorithm]=np.zeros((repeat,T),float)
        avg_regret_sum[algorithm]=np.zeros(T,float)
        std_R[algorithm]=np.zeros(T,float)
        
    print('repeat',i)
    seed=i
    Env=linear_Env(seed,d,N,K,L,eps)
    delta=Env.delta()
    alg_MW=MaxWeight(seed,Env.x, N, K, L, T, Env.theta)
    alg_UCB=UCB_QMB(seed,Env.x, N, K, L, T)
    alg_TS=TS_QMB(seed,Env.x, N, K, L, T)
    alg_ETC_GS=ETC_GS(seed,Env.x, N, K, L, T,delta)
    alg_DAM_UCB=DAM_UCB(seed,Env.x,N,K,L,T,eps)
    alg_MW_UCB=MaxWeight_UCB(seed,Env.x,N,K,L,T)
    alg_Q_UCB=Q_UCB(seed,Env.x,N,K,L,T)

    algorithms=[alg_MW,alg_UCB,alg_TS,alg_ETC_GS,alg_DAM_UCB,alg_MW_UCB,alg_Q_UCB]


    for algorithm in algorithms:
        name=algorithm.name()
        Q_sum[name]=[]
        exp_reward[name]=[]              
        Env_dict[name]=linear_Env(seed,d,N,K,L,eps)

    for algorithm in algorithms:
        algorithm.reset()
        name=algorithm.name()
        print(name)
        for t in tqdm((np.array(range(T))+1)):
            if t==1:
                Q[name]=np.zeros(N)
                algorithm.run(t,np.zeros(K),np.zeros(N))
                if name!='MaxWeight': #for regret
                    alg_MW.run(t,np.zeros(K),np.zeros(N))
            else:
                algorithm.run(t,index[name], Q[name])
                if name!='MaxWeight': #for regret
                    alg_MW.run(t,index[name], Q[name])

            S=algorithm.offer()
            S_MW=alg_MW.offer()
            exp_reward[name].append(Env_dict[name].exp_reward(S,Q[name]))
            oracle_reward[name].append(Env_dict[name].exp_reward(S_MW,Q[name]))
            Q_sum[name].append(sum(Q[name]))
            index[name],Q[name]=Env_dict[name].observe(S)


 
    for algorithm in algorithms:
        name=algorithm.name()
        Q_sum_cum=np.cumsum(Q_sum[name])
        indexes = np.arange(1, len(Q_sum[name]) + 1)
        Q_sum_cum_avg=Q_sum_cum/ indexes
        Q_sum_list[name][i,:]=Q_sum_cum_avg
        avg_Q_sum[name]+=Q_sum_cum_avg

        reg=np.array(oracle_reward[name])-np.array(exp_reward[name])
        regret_sum=np.cumsum(reg)
        regret_sum_list[name][i,:]=regret_sum
        avg_regret_sum[name]+=regret_sum  


        filename_1=name+'T'+str(T)+'d'+str(d)+'N'+str(N)+'K'+str(K)+'L'+str(L)+'repeat'+str(i)+'eps'+str(eps)+'Q.txt'
        with open('./result/'+filename_1, 'wb') as f:
            pickle.dump(Q_sum_cum_avg, f)
            f.close()

        filename_1=name+'T'+str(T)+'d'+str(d)+'N'+str(N)+'K'+str(K)+'L'+str(L)+'repeat'+str(i)+'eps'+str(eps)+'R.txt'
        with open('./result/'+filename_1, 'wb') as f:
            pickle.dump(regret_sum, f)
            f.close()

    Q_sum.clear()
    avg_Q_sum.clear()
    Q_sum_list.clear()
    std.clear()
    index.clear()
    Q.clear()
    Env_dict.clear()
    S.clear()
    exp_reward.clear()
    avg_regret_sum.clear()
    oracle_reward.clear()
    regret_sum_list.clear()
    std_R.clear()

def run_multiprocessing(T, repeat, d, N, K, L, eps):
    Path("./result").mkdir(parents=True, exist_ok=True)

        
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(run, [( T, repeat, d, N, K, L, i,eps) for i in range(repeat)])

    pool.close()
    pool.join()

if __name__=='__main__':

    L=2  
    T=20000
    repeat=10
    eps=0.1
    d=2
    N=4
    K=2 #2, 3
    print(N,K,L, eps)
    run_multiprocessing(T,repeat,d,N,K,L,eps)
   


