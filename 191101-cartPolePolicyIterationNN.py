import sys
from bisect import bisect_left
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

class Net(nn.Module):
    def __init__(self, lIn=4, nInner=2, width=100, lOut=2):
        super().__init__() #??
        self.nInner=nInner
        self.fcIn=nn.Linear(lIn, width)
        self.fcInner=[nn.Linear(width, width) for _ in range(self.nInner)]
        self.fcOut=nn.Linear(width, lOut)

        self.sm=nn.Softmax(dim=1)
        self.d1d=nn.Dropout(p=0.1)

    def forwardTest(self, x):
        x=F.relu(self.fcIn(x))
        for i in range(self.nInner):
            x=F.relu(self.fcInner[i](x))
        x=self.fcOut(x)
        return x

    def forwardTrain(self, x):
        x=F.relu(self.d1d(self.fcIn(x)))
        for i in range(self.nInner):
            x=F.relu(self.d1d(self.fcInner[i](x)))
        x=self.d1d(self.fcOut(x))
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

def compute1dIndex(index,shape):
    d=len(shape)
    index1d=0
    multiplier=1
    for i in range(-1,-len(shape)-1,-1):
        index1d+=multiplier*index[i]
        multiplier*=shape[i]
    return index1d

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
        actV=sharedArgs['net'].forwardTest(ttobs)
        actP=sharedArgs['net'].sm(actV)[0]
        act=sco.actionFromProbabilities(actP)
        index=compute1dIndex((episode,t),sharedArgs['actIndShape'])
        sharedArgs['actInd'][index]=act
        index=compute1dIndex((episode,t,0),sharedArgs['obsShape'])
        for i in range(sharedArgs['obsShape'][-1]):
            sharedArgs['obs'][index+i]=obs[i]
        index=compute1dIndex((episode,t),sharedArgs['rewShape'])
        sharedArgs['rew'][index]=int(rew)
        obs,rew,fin,inf=env.step(act)
        if fin:
            index=compute1dIndex((episode,),sharedArgs['durShape'])
            sharedArgs['dur'][index]=t+1
            break

def runEpisodeSet(env,nEpisodes,maxEpisodeLength,policyNet,policyLossFn,valueNet,
                  valueLossFn,sco,discountFactor=1.,train=True,render=True,
                  learningRate=1e-2,rngSeed=None,nProcs=mp.cpu_count()//2):
    err=sys.stderr
    rng=np.random.RandomState(seed=rngSeed)

    #setup arrays for parallel trj runs
    actIndShape=(nEpisodes,maxEpisodeLength)
    actIndRaw=mp.RawArray('i',actIndShape[0]*actIndShape[1])
    rewShape=(nEpisodes,maxEpisodeLength)
    rewRaw=mp.RawArray('i',rewShape[0]*rewShape[1])
    obsShape=(nEpisodes,maxEpisodeLength,4)
    obsRaw=mp.RawArray('d',obsShape[0]*obsShape[1]*obsShape[2])
    durShape=(nEpisodes,)
    durRaw=mp.RawArray('i',durShape[0])
    initArgs=(policyNet,
              actIndRaw,actIndShape,
              rewRaw,rewShape,
              obsRaw,obsShape,
              durRaw,durShape)

    #launch parallel trj runs to get data
    pool=mp.Pool(nProcs,initializer=loadSharedArgs,initargs=initArgs)
    args=[(env,sco,episode,nEpisodes,maxEpisodeLength,
           rng.randint(np.iinfo(np.int32).max)) for episode in range(nEpisodes)]
    pool.starmap(runEpisode,args)
    pool.close()
    pool.join()

    #load parallel trj data into np arrays
    actIndArr=np.frombuffer(actIndRaw,dtype=np.int32).reshape(actIndShape)
    rewArr=np.frombuffer(rewRaw,dtype=np.int32).reshape(rewShape)
    obsArr=np.frombuffer(obsRaw).reshape(obsShape)
    durArr=np.frombuffer(durRaw,dtype=np.int32).reshape(durShape)

    if train is True: #training can only happen after have trj return
        valueOptimizer=torch.optim.SGD(valueNet.parameters(), lr=learningRate)
        valueOptimizer.zero_grad()
        policyOptimizer=torch.optim.Adam(policyNet.parameters(), lr=learningRate)
        policyOptimizer.zero_grad()
        #roll episodes into mini-batch to "vectorize"
        ret=[]
        obs=[]
        actInd=[]
        for episode in range(nEpisodes):
            ret+=[rewArr[episode,t:durArr[episode]].sum() 
                  for t in range(durArr[episode])]
            obs+=list(obsArr[episode,:durArr[episode]])
            actInd+=list(actIndArr[episode,:durArr[episode]])
        ret=torch.tensor(ret)
        obs=torch.tensor(obs).float()
        actInd=torch.tensor(actInd).long()
        #update value estimator net & compute val estmates
        val=valueNet.forwardTest(obs)
        valueLoss=valueLossFn(val,ret.float()[:,None])
        valueLoss.backward()
        valueOptimizer.step()
        val=valueNet.forwardTest(obs)[:,0]
        #compute losses & update NN
        actV=policyNet.forwardTest(obs)
        policyLoss=policyLossFn(actV,actInd)*(ret-val)/nEpisodes
        policyLoss=policyLoss.mean()
        policyLoss.backward()
        policyOptimizer.step()
    return policyNet,valueNet,actIndArr,rewArr,obsArr,durArr

def main(nInnerPolicy=1,widthPolicy=32,nInnerValue=1,widthValue=32,
         nTrainingEpochs=100,nTrainingEpisodes=500,maxTrainingEpisodeLength=500,
         nTestEpochs=10,nTestEpisodes=1,maxTestEpisodeLength=500,discountFactor=1.0,
         envSeed=None,render=False,rngSeed=None,
         fOutNmTemplate='cartPoleRewardDistribution'):
    err=sys.stderr
    env=gym.make('CartPole-v1')
    print(env.seed(envSeed),file=err)
    policyNet=Net(lIn=4,nInner=nInnerPolicy,width=widthPolicy,lOut=2)
    policyLossFn=nn.CrossEntropyLoss(reduction='none') #need to weight by ret before reducing
    valueNet=Net(lIn=4,nInner=nInnerValue,width=widthValue,lOut=1)
    valueLossFn=nn.MSELoss()
    sco=StochasticChooseOutput(seed=rngSeed)
    print('nInnerPolicy, widthPolicy: {}, {}'.format(nInnerPolicy,widthPolicy),file=err)
    print('nInnerValue, widthValue: {}, {}'.format(nInnerValue,widthValue),file=err)
    print('#ep\ttFmean\tmin\tmax')

    for n in range(0,nTrainingEpochs):
        policyNet,valueNet,actArr,rewArr,obsArr,durArr= \
                runEpisodeSet(env,nTrainingEpisodes,maxTrainingEpisodeLength,
                              policyNet,policyLossFn,valueNet,valueLossFn,sco,
                              train=True,render=render)
        #basic stats
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()))
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()),file=err)
        ##distributions
        #fOutNm=fOutNmTemplate+'Epoch{}.out'.format(n)
        #with open(fOutNm,'w') as f:
        #    durVals,durNums=np.unique(durArr,return_counts=True)
        #    print('#dur/R\tnum\tP[dur/R]',file=f)
        #    for i in range(len(durVals)):
        #        print('{}\t{}\t{}'.format(durVals[i],durNums[i],durNums[i]/nTrainingEpisodes),file=f)

    print('\ntest:\n')
    for n in range(0,nTestEpochs):
        policyNet,valueNet,actArr,rewArr,obsArr,durArr= \
                runEpisodeSet(env,nTestEpisodes,maxTestEpisodeLength,
                              policyNet,policyLossFn,valueNet,valueLossFn,sco,
                              train=False,render=render)
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()))
        print('{}\t{}\t{}\t{}'.format(n,durArr.mean(),durArr.min(),durArr.max()),file=err)

if __name__=='__main__':
    main()
