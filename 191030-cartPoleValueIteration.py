import sys
from math import floor,pi
import numpy as np
from numba import jit
import gym

#lifted from python3 math.isclose()
@jit(nopython=True,cache=True)
def isclose(a,b,relTol,absTol):
  mab=max(abs(a),abs(b))
  mabAbsTol=max(relTol*mab,absTol)
  dif=abs(a-b)
  return dif<=mabAbsTol

#0-161: in bounds
#-1: out of bounds
@jit(nopython=True,cache=True)
def cgObsToState(obs):
    x,xDot,theta,thetaDot=obs
    oneDeg=2.0*pi/360*1.
    sixDeg=2.0*pi/360*6.
    maxDeg=2.0*pi/360*12.
    fiftyDeg=2.0*pi/360*50.
    state=0
    if x<-2.4 or x>2.4 or theta<-maxDeg or theta>maxDeg:
        return -1
    if x<-0.8:
        state=0
    elif x<0.8:
        state=1
    else:
        state=2

    if xDot<-0.5:
        pass
    elif xDot<0.5:
        state+=3
    else:
        state+=6

    if theta<-sixDeg:
        pass
    elif theta<-oneDeg:
        state+=9
    elif theta<0:
        state+=18
    elif theta<oneDeg:
        state+=27
    elif theta<sixDeg:
        state+=36
    else:
        state+=45

    if thetaDot<-fiftyDeg:
        pass
    elif thetaDot<fiftyDeg:
        state+=54
    else:
        state+=108
    return state

#@jit(nopython=True,cache=True)
def expandStateToObs(s):
    if s==162: return 3,3,6,3
    elif s>162 or s<0: exit(1)
    nThetaDot=s//54
    s2=s-nThetaDot*54
    nTheta=s2//9
    s3=s2-nTheta*9
    nXDot=s3//3
    s4=s3-nXDot*3
    nX=s4
    #print(s,nX,nXDot,nTheta,nThetaDot,file=sys.stderr)
    #bin mids:
    if nX==0:
        x=(-2.4+0.8)/2.
    elif nX==1:
        x=0.
    elif nX==2:
        x=(2.4-0.8)/2.
    
    if nTheta==0:
        theta=-9.
    elif nTheta==1:
        theta=-3.5
    elif nTheta==2:
        theta=-0.5
    elif nTheta==3:
        theta=0.5
    elif nTheta==4:
        theta=3.4
    elif nTheta==5:
        theta=9.

    #guess velocity bins
    if nXDot==0:
        xDot=-1.
    elif nXDot==1:
        xDot=0
    elif nXDot==2:
        xDot=1.

    if nThetaDot==0:
        thetaDot=-10.
    elif nThetaDot==1:
        thetaDot=0.
    elif nThetaDot==2:
        thetaDot=10.

    return x,xDot,theta,thetaDot

#@jit(nopython=True,cache=True)
def updateTransitionProb(actArr,obsArr,psa,n):
#def updateTransitionProb(actArr,obsArr,psa,n,value):
    ns,na,_=psa.shape
    for s in range(ns):
        for a in range(na):
            psa[s,a]=psa[s,a]*n[s,a]
    trjLen=len(actArr)
    sArr=np.zeros(trjLen,dtype=np.int32)
    for i in range(trjLen):
        sArr[i]=cgObsToState(obsArr[i])
    #print('s:{}'.format(sArr),file=sys.stderr)
    #print('v(s):{}'.format(value[sArr]),file=sys.stderr)
    for t in range(trjLen-1):
        st=sArr[t]
        at=actArr[t]
        stPlus1=sArr[t+1]
        psa[st,at,stPlus1]+=1.
        n[st,at]+=1.
    #final (failing) step. important for propagation of failure
    st=sArr[-1]
    at=actArr[-1]
    stPlus1=-1
    psa[st,at,stPlus1]+=1.
    n[st,at]+=1.
    for s in range(ns):
        for a in range(na):
            if n[s,a]>0.:
                psa[s,a]/=n[s,a]
            else:
                psa[s,a]=1./ns
    return psa,n

@jit(nopython=True,cache=True)
def rlValueIteration(reward,discount,psa,maxIter):
    ns,=reward.shape
    valueOld=reward.copy()
    valueNew=np.zeros(ns)
    for i in range(0,maxIter):
        for s in range(0,ns-1):
            valueNew[s]=reward[s]+discount*(psa[s]@valueOld).max()
        valueNew[-1]=reward[-1] #final state: no more moves
        for j in range(0,ns):
            if isclose(valueNew[j],valueOld[j],1e-9,0.) is False:
                break
        valueOld=np.copy(valueNew)
        if j==ns-1:
            endVia='allclose'
            break
    if i==maxIter-1:
        endVia='maxIter'
    return valueNew,endVia

@jit(nopython=True,cache=True)
def policyArrFromValue(value,psa):
    ns,na,_=psa.shape
    policyArr=np.zeros(ns,dtype=np.int32)
    for s in range(0,ns):
        policyArr[s]=np.argmax(psa[s]@value)
    return policyArr

@jit(nopython=True,cache=True)
def policyViaValueIteration(obs,policyArr):
    s=cgObsToState(obs)
    return policyArr[s]

@jit(nopython=True,cache=True)
def transitionProbChange(psaOld,psaNew):
    deltaPsa=psaNew-psaOld
    return (deltaPsa*deltaPsa).sum()

def runEpisodeSet(env,rng,nEpisodes,maxEpisodeLength,psa,n,reward,policy,policyArr=None,pGreed=0.9,valueIterateEvery=5,discountFactor=1.,valueIterationMaxIter=int(1e4),train=True,render=True):
    err=sys.stderr
    for episode in range(nEpisodes):
        actArr=-np.ones(maxEpisodeLength,dtype=np.int32)
        obsArr=np.zeros((maxEpisodeLength,4))
        rewTot=0
        obs=env.reset()
        for t in range(maxEpisodeLength):
            if render:
                env.render()
            if rng.uniform()<pGreed:
                act=policy(obs,policyArr) #choose action according to policy
            else:
                act=env.action_space.sample() #choose random action
            actArr[t]=act
            obsArr[t]=obs
            obs,rew,fin,inf=env.step(act)
            rewTot+=rew
            if fin:
                actArr=actArr[:t+1]
                obsArr=obsArr[:t+1]
                break
        if train is True:
            psaOld=psa.copy()
            psa,n=updateTransitionProb(actArr,obsArr,psa,n)
            #psa,n=updateTransitionProb(actArr,obsArr,psa,n,value)
            if episode%valueIterateEvery==0:
                value,endVia=rlValueIteration(reward,discountFactor,psa,valueIterationMaxIter)
                policyArr=policyArrFromValue(value,psa)
                print('#s\tv(s)\tx\txDot\ttheta\tthetaDot',file=err)
                for s in range(len(value)):
                    print('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(s,value[s],*expandStateToObs(s)),file=err)
                print('',file=err)
                #print(value,file=err)
                #print(endVia,file=err)
            print('{}\t{}\t{}'.format(episode,t,transitionProbChange(psaOld,psa)))
        else:
            print('{}\t{}\t0'.format(episode,t))
        #print('act: {}'.format(actArr),file=err)
    if train is True:
        return psa,n,policyArr
    else:
        return psa,n,None

def main(nPreTrainingEpisodes=10,nTrainingEpisodes=500,maxTrainingEpisodeLength=500,nTestEpisodes=100,maxTestEpisodeLength=500,pGreed=0.9,valueIterateEvery=1,discountFactor=0.99,valueIterationMaxIter=int(1e4),nTotalStates=163,rngSeed=None,envSeed=None):
    err=sys.stderr
    env=gym.make('CartPole-v1')
    print(env.seed(envSeed),file=err)
    reward=np.ones(nTotalStates)
    reward[-1]=0.
    #reward-=1.
    rng=np.random.RandomState(seed=rngSeed)
    psa=np.ones((nTotalStates,2,nTotalStates))/nTotalStates #uniform
    n=np.zeros((nTotalStates,2))
    print('#ep\ttFin\tdelPsa')
#def runEpisodeSet(nEpisodes,maxEpisodeLength,psa,n,reward,value,policy,pGreed=0.8,valueIterateEvery=5,discountFactor=0.99,valueIterationMaxIter=int(1e3)):
    policy=lambda obs,policyArr: env.action_space.sample()
    psa,n,policyArr=runEpisodeSet(env,rng,nPreTrainingEpisodes,maxTrainingEpisodeLength,psa,n,reward,policy,policyArr=None,pGreed=0.,train=True,render=False)
    print(policyArr.sum(),file=err)
    policy=lambda obs,policyArr: policyViaValueIteration(obs,policyArr)
    psa,n,policyArr=runEpisodeSet(env,rng,nTrainingEpisodes,maxTrainingEpisodeLength,psa,n,reward,policy,policyArr=policyArr,pGreed=pGreed,train=True,render=False)
    print(policyArr.sum(),file=err)
    psa,n,_=runEpisodeSet(env,rng,nTestEpisodes,maxTestEpisodeLength,psa,n,reward,policy,policyArr=policyArr,pGreed=1.,train=False,render=True)

if __name__=='__main__':
    main()
