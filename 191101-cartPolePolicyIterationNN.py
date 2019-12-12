import sys
from math import floor,pi
from bisect import bisect_left
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

class Net(nn.Module):
    def __init__(self, nInner=2, width=100):
        super().__init__() #??
        self.nInner=nInner
        self.fcIn=nn.Linear(4, width)
        self.fcInner=[nn.Linear(width, width) for _ in range(self.nInner)]
        self.fcOut=nn.Linear(width, 2)

        self.sm=nn.Softmax(dim=1)
        self.d1d=nn.Dropout(p=0.1)

    def forwardTest(self, x):
        x=F.relu(self.fcIn(x))
        for i in range(self.nInner):
            x=F.relu(self.fcInner[i](x))
        x=self.sm(self.fcOut(x))
        return x

    def forwardTrain(self, x):
        x=F.relu(self.d1d(self.fcIn(x)))
        for i in range(self.nInner):
            x=F.relu(self.d1d(self.fcInner[i](x)))
        x=self.sm(self.d1d(self.fcOut(x)))
        return x

class StochasticChooseOutput():
    def __init__(self, seed=None):
        self.rng=np.random.RandomState(seed=seed)

    def rollUniform(self):
        return self.rng.uniform()

    def actionFromProbabilities(self, p):
        l=len(p)
        q=[p[:i+1].sum() for i in range(l)]
        rand=self.rollUniform()
        a=bisect_left(q, rand)
        return a

def lossFn(act, ret, i):
    return -ret*torch.log(act[i])

def runEpisodeSet(env,nEpisodes,maxEpisodeLength,net,sco,discountFactor=1.,train=True,render=True,learningRate=1e-2):
    err=sys.stderr
    actIndArr=np.zeros((nEpisodes,maxEpisodeLength),dtype=np.int32)
    rewArr=np.zeros((nEpisodes,maxEpisodeLength),dtype=np.int32)
    obsArr=np.zeros((nEpisodes,maxEpisodeLength,4))
    durArr=np.ones(nEpisodes,dtype=np.int32)*maxEpisodeLength
    for episode in range(nEpisodes):
        obs=env.reset()
        rew=1
        for t in range(maxEpisodeLength):
            if render:
                env.render()
            ttobs=torch.tensor(obs)[None,:].float()
            actP=net.forwardTrain(ttobs)[0]
            act=sco.actionFromProbabilities(actP)
            actIndArr[episode,t]=act
            obsArr[episode,t]=obs
            rewArr[episode,t]=rew
            obs,rew,fin,inf=env.step(act)
            if fin:
                durArr[episode]=t+1
                break

    if train is True: #wait till end to determine return for
        net.zero_grad()
        for episode in range(nEpisodes): #each action
            for t in range(durArr[episode]):
                ret=rewArr[episode,t:durArr[episode]].sum()
                ttobs=torch.tensor(obsArr[episode,t])[None,:].float()
                act=net.forwardTest(ttobs)[0]
                #act=net.forwardTrain(ttobs)[0]
                loss=lossFn(act,ret,actIndArr[episode,t])/nEpisodes
                print(episode,t,ret,act,loss.item(),file=sys.stderr)
                loss.backward() #accumulate grad
        for p in net.parameters(): #update based on accumulated
            p.data.sub_(learningRate*p.grad.data) #...gradients

    return net,actIndArr,rewArr,obsArr,durArr

def main(nInner=1,width=50,nTrainingEpochs=100,nTrainingEpisodes=500,maxTrainingEpisodeLength=500,nTestEpochs=10,nTestEpisodes=1,maxTestEpisodeLength=500,discountFactor=1.0,envSeed=None,render=False,rngSeed=None):
    err=sys.stderr
    env=gym.make('CartPole-v1')
    print(env.seed(envSeed),file=err)
    net=Net(nInner=nInner,width=width)
    sco=StochasticChooseOutput(seed=rngSeed)
    print('#ep\ttFin.mean min\tmax')

    for n in range(0,nTrainingEpochs):
        net,actArr,rewArr,obsArr,durArr=runEpisodeSet(env,nTrainingEpisodes,maxTrainingEpisodeLength,net,sco,train=True,render=render)
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()))

    print('test:')
    for n in range(0,nTestEpochs):
        net,actArr,rewArr,obsArr,durArr=runEpisodeSet(env,nTrainingEpisodes,maxTrainingEpisodeLength,net,sco,train=False,render=render)
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()))

if __name__=='__main__':
    main()
