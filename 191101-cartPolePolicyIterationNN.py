import sys
from bisect import bisect_left
import multiprocessing as mp
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

def compute1dIndex(index,shape):
    d=len(shape)
    index1d=0
    multiplier=1
    for i in range(-1,-len(shape)-1,-1):
        index1d+=multiplier*index[i]
        multiplier*=shape[i]
    return index1d

def runEpisode(env,sco,episode,nEpisodes,maxEpisodeLength,seed=None,render=False):
    rng=np.random.RandomState(seed=seed)
    envSeed=rng.randint(np.iinfo(np.int).max)
    env.seed(envSeed)
    scoSeed=rng.randint(np.iinfo(np.int32).max)
    sco.rng.seed(scoSeed)
    obs=env.reset()
    rew=1
    for t in range(maxEpisodeLength):
        if render:
            env.render()
        ttobs=torch.tensor(obs)[None,:].float()
        actP=sharedArgs['net'].forwardTrain(ttobs)[0]
        act=sco.actionFromProbabilities(actP)
        #actIndArr[episode,t]=act
        index=compute1dIndex((episode,t),sharedArgs['actIndShape'])
        sharedArgs['actInd'][index]=act
        #obsArr[episode,t]=obs
        index=compute1dIndex((episode,t,0),sharedArgs['obsShape'])
        for i in range(sharedArgs['obsShape'][-1]):
            sharedArgs['obs'][index+i]=obs[i]
        #rewArr[episode,t]=rew
        index=compute1dIndex((episode,t),sharedArgs['rewShape'])
        sharedArgs['rew'][index]=int(rew)
        obs,rew,fin,inf=env.step(act)
        if fin:
            #durArr[episode]=t+1
            index=compute1dIndex((episode,),sharedArgs['durShape'])
            sharedArgs['dur'][index]=t+1
            break
    #return actIndArr,obsArr,rewArr,durArr

sharedArgs={}
def loadSharedArgs(net,
                   actIndRaw,actIndShape,
                   rewRaw,rewShape,
                   obsRaw,obsShape,
                   durRaw,durShape):
    sharedArgs['net']=net
    sharedArgs['actInd']=actIndRaw
    sharedArgs['actIndShape']=actIndShape
    sharedArgs['rew']=rewRaw
    sharedArgs['rewShape']=rewShape
    sharedArgs['obs']=obsRaw
    sharedArgs['obsShape']=obsShape
    sharedArgs['dur']=durRaw
    sharedArgs['durShape']=durShape

def runEpisodeSet(env,nEpisodes,maxEpisodeLength,net,sco,discountFactor=1.,train=True,render=True,learningRate=1e-2,rngSeed=None,nProcs=mp.cpu_count()//2):
    err=sys.stderr
    rng=np.random.RandomState(seed=rngSeed)
    #actIndArr=np.zeros((nEpisodes,maxEpisodeLength),dtype=np.int32)
    #rewArr=np.zeros((nEpisodes,maxEpisodeLength),dtype=np.int32)
    #obsArr=np.zeros((nEpisodes,maxEpisodeLength,4))
    #durArr=np.ones(nEpisodes,dtype=np.int32)*maxEpisodeLength

    actIndShape=(nEpisodes,maxEpisodeLength)
    actIndRaw=mp.RawArray('i',actIndShape[0]*actIndShape[1])
    #np.copyto(actIndRaw,actIndArr)
    rewShape=(nEpisodes,maxEpisodeLength)
    rewRaw=mp.RawArray('i',rewShape[0]*rewShape[1])
    #np.copyto(rewRaw,rewArr)
    obsShape=(nEpisodes,maxEpisodeLength,4)
    obsRaw=mp.RawArray('d',obsShape[0]*obsShape[1]*obsShape[2])
    #np.copyto(obsRaw,obsArr)
    durShape=(nEpisodes,)
    durRaw=mp.RawArray('i',durShape[0])
    #np.copyto(durRaw,durArr)
    initArgs=(net,
              actIndRaw,actIndShape,
              rewRaw,rewShape,
              obsRaw,obsShape,
              durRaw,durShape)

    pool=mp.Pool(nProcs,initializer=loadSharedArgs,initargs=initArgs)
    args=[(env,sco,episode,nEpisodes,maxEpisodeLength,
           rng.randint(np.iinfo(np.int32).max)) for episode in range(nEpisodes)]
    pool.starmap(runEpisode,args)
    pool.close()
    pool.join()
    actIndArr=np.frombuffer(actIndRaw,dtype=np.int32).reshape(actIndShape)
    rewArr=np.frombuffer(rewRaw,dtype=np.int32).reshape(rewShape)
    obsArr=np.frombuffer(obsRaw).reshape(obsShape)
    durArr=np.frombuffer(durRaw,dtype=np.int32).reshape(durShape)
    if train is True: #wait till end to determine return
        net.zero_grad() #...for each action
        for episode in range(nEpisodes):
            for t in range(durArr[episode]):
                ret=rewArr[episode,t:durArr[episode]].sum()
                ttobs=torch.tensor(obsArr[episode,t])[None,:].float()
                act=net.forwardTest(ttobs)[0]
                #act=net.forwardTrain(ttobs)[0]
                loss=lossFn(act,ret,actIndArr[episode,t])/nEpisodes
                print(episode,t,ret,act,loss.item(),file=err)
                loss.backward() #accumulate grad
        for p in net.parameters(): #update based on accumulated
            p.data.sub_(learningRate*p.grad.data) #...gradients
    return net,actIndArr,rewArr,obsArr,durArr

def main(nInner=2,width=100,nTrainingEpochs=100,nTrainingEpisodes=500,maxTrainingEpisodeLength=500,nTestEpochs=10,nTestEpisodes=1,maxTestEpisodeLength=500,discountFactor=1.0,envSeed=None,render=False,rngSeed=None):
    err=sys.stderr
    env=gym.make('CartPole-v1')
    print(env.seed(envSeed),file=err)
    net=Net(nInner=nInner,width=width)
    sco=StochasticChooseOutput(seed=rngSeed)
    print('#ep\ttFmean\tmin\tmax')

    for n in range(0,nTrainingEpochs):
        net,actArr,rewArr,obsArr,durArr=runEpisodeSet(env,nTrainingEpisodes,maxTrainingEpisodeLength,net,sco,train=True,render=render)
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()))
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()),file=err)

    print('test:')
    for n in range(0,nTestEpochs):
        net,actArr,rewArr,obsArr,durArr=runEpisodeSet(env,nTrainingEpisodes,maxTrainingEpisodeLength,net,sco,train=False,render=render)
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()))
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()),file=err)

if __name__=='__main__':
    main()
