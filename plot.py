from Environment import *
from Algorithms import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import matplotlib.gridspec as gridspec


def plot(T,repeat,d,N,K,L,eps):

    avg_Q_sum=dict()
    Q_sum_list=dict()
    std=dict()
    Q=dict()
    y=dict()
    S=dict()

    exp_reward=dict()
    avg_regret_sum=dict()
    regret_sum_list=dict()
    std_R=dict()
    regret=dict()
    for algorithm in ['MaxWeight','UCB_QMB','TS_QMB','ETC-GS','Q-UCB','DAM-UCB','MaxWeight-UCB']:
        Q_sum_list[algorithm]=np.zeros((repeat,T),float)
        avg_Q_sum[algorithm]=np.zeros(T,float)
        std[algorithm]=np.zeros(T,float)

        exp_reward[algorithm]=[]    
        regret_sum_list[algorithm]=np.zeros((repeat,T),float)
        avg_regret_sum[algorithm]=np.zeros(T,float)
        std[algorithm]=np.zeros(T,float)
        
    algorithms=['ETC-GS','UCB_QMB','Q-UCB','TS_QMB','DAM-UCB','MaxWeight','MaxWeight-UCB']
    gs = gridspec.GridSpec(1,2) 
    fig = plt.figure(figsize=(18, 7))
    for algorithm in  algorithms:
        name=algorithm
        for i in range(repeat):
            filename_1=name+'T'+str(T)+'d'+str(d)+'N'+str(N)+'K'+str(K)+'L'+str(L)+'repeat'+str(i)+'eps'+str(eps)+'Q.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            Q[name]=objects[0]
            Q_sum_list[name][i,:]=objects[0]
            avg_Q_sum[name]+=objects[0]  


            filename_1=name+'T'+str(T)+'d'+str(d)+'N'+str(N)+'K'+str(K)+'L'+str(L)+'repeat'+str(i)+'eps'+str(eps)+'R.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            regret[name]=objects[0]
            regret_sum_list[name][i,:]=objects[0]
            avg_regret_sum[name]+=objects[0]  

        Q[name]=avg_Q_sum[name]/repeat
        std[name]=np.std(Q_sum_list[name],axis=0)
        regret[name]=avg_regret_sum[name]/repeat
        std_R[name]=np.std(regret_sum_list[name],axis=0)


    T_p=int(T/10)
    T_p2=int(T/57)
    T_p3=int(T/21)
    size=30
    ax = fig.add_subplot(gs[0, 0])
    ax.tick_params(labelsize=size)
    plt.rc('legend',fontsize=size)
    ax.yaxis.get_offset_text().set_fontsize(size)
    ax.xaxis.get_offset_text().set_fontsize(size)
    plt.gcf().subplots_adjust(bottom=0.20)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    color=['gray','royalblue','gold','limegreen','lightsalmon','violet','slateblue']
    marker_list=['P','<','v','D','^','o','s']

    for i, algorithm in enumerate(algorithms):
        name=algorithm
        if name=='TS_QMB':
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),Q[name],color=col, marker=mark, label='TS-QMB (Algorithm 2)', markersize=11,markevery=T_p,zorder=10)
            ax.errorbar(range(T), Q[name], yerr=std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=10)
        elif name=='UCB_QMB':
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),Q[name],color=col, marker=mark, label='UCB-QMB (Algorithm 1)', markersize=10,markevery=T_p,zorder=11)
            ax.errorbar(range(T),Q[name], yerr=std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=11)
        elif name=='MaxWeight':
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),Q[name],color=col, marker=mark, label='Oracle (MaxWeight)', markersize=12,markevery=T_p,zorder=9)
            ax.errorbar(range(T),Q[name], yerr=std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=9)    
        elif name=='ETC-GS':
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),Q[name],color=col, linestyle='dashed',label=name,zorder=8-i)
            ax.errorbar(range(T), Q[name], yerr=std[name]/np.sqrt(repeat),linestyle='dashed', color=col, errorevery=T_p, capsize=6,zorder=8-i)
        else:
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),Q[name],color=col, marker=mark, label=name, markersize=12,markevery=T_p,zorder=8-i)
            ax.errorbar(range(T), Q[name], yerr=std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=8-i)

    plt.title('Average Queue Length',fontsize=size)
    plt.xlabel('Time step '+r'$t$',fontsize=size)
    plt.ylabel(r'$\mathcal{Q}(t)$',fontsize=size)
    plt.ylim([0, 200])

    T_p=int(T/10)
    ax = fig.add_subplot(gs[0, 1])
    ax.tick_params(labelsize=size)
    plt.rc('legend',fontsize=size)
    ax.yaxis.get_offset_text().set_fontsize(size)
    ax.xaxis.get_offset_text().set_fontsize(size)
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.tight_layout()

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    for i, algorithm in enumerate(algorithms):
        name=algorithm
        if name=='TS_QMB':
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),regret[name],color=col, marker=mark, label='TS-QMB (Algorithm 2)', markersize=12,markevery=T_p,zorder=10)
            ax.errorbar(range(T), regret[name], yerr=std_R[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=10)
        elif name=='UCB_QMB':
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),regret[name],color=col, marker=mark, label='UCB-QMB (Algorithm 1)', markersize=12,markevery=T_p,zorder=11)
            ax.errorbar(range(T), regret[name], yerr=std_R[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=11)
        elif name=='MaxWeight':
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),regret[name],color=col, marker=mark, label='Oracle (MaxWeight)', markersize=12,markevery=T_p,zorder=9)
            ax.errorbar(range(T),regret[name], yerr=std_R[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=9)    
        elif name=='ETC-GS':
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),regret[name],color=col, linestyle='dashed',label=name,zorder=8-i)
            ax.errorbar(range(T), regret[name], yerr=std_R[name]/np.sqrt(repeat), linestyle='dashed', color=col, errorevery=T_p, capsize=6,zorder=8-i)  
        else:
            col=color[i]
            mark=marker_list[i]
            ax.plot(range(T),regret[name],color=col, marker=mark, label=name, markersize=12,markevery=T_p,zorder=8-i)
            ax.errorbar(range(T), regret[name], yerr=std_R[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=8-i)

    plt.title('Regret',fontsize=size)
    plt.xlabel('Time step '+r'$t$',fontsize=size)
    plt.ylabel(r'$\mathcal{R}(t)$',fontsize=size)
    plt.ylim([0, 200000])


    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    labels = [labels[0], labels[1], labels[2], labels[3],labels[4],labels[5],labels[6]]
    lines=[lines[0],lines[1],lines[2],lines[3],lines[4],lines[5],lines[6]]
    fig.legend(lines, labels, loc='upper center', ncol=4,bbox_to_anchor=(0.5, 1.23))
    plt.tight_layout()

    plt.savefig('./plot/T'+str(T)+'d'+str(d)+'N'+str(N)+'K'+str(K)+'L'+str(L)+'repeat'+str(repeat)+'.pdf', bbox_inches = "tight")
    plt.show()  


if __name__=='__main__':
    Path("./plot").mkdir(parents=True, exist_ok=True)
    d=2
    L=2
    N=4
    K=2 #2, 3
    T=20000
    eps=0.1

    repeat=10
    plot(T,repeat,d,N,K,L,eps)
   


