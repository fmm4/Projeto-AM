import neurolab as nl
import numpy as np
import scipy as sp
from sklearn import svm
import csv
import time
import random
import math 
import itertools
import scipy.stats
import ipdb

# Extraido do exercicio de dissimilaridade

def delta(i,j):
    if i != j:
        return 1
    else:
        return 0

def dissimilarity(x1,x2):
    return sum([delta(x1[i],x2[i]) for i in range(len(x1)-1)])
        
def calculate_dmatrix(matrix):
    matrix_heigth = len(matrix)
    matrix_width = len(matrix[0][:-1])
    posi_lines = []
    nega_lines = []
    dmatrix = []
    for i in range(matrix_heigth):
        new_line = []
        for j in range(matrix_heigth):
            new_line.append(dissimilarity(matrix[i],matrix[j]))
        dmatrix.append(new_line)
    return dmatrix

#########################################################3

def calculateProbability(type,numbers,slot):
    if (type == 'P'):
        return sum((x[slot]*(x[slot]+1))/2 for x in numbers)/len(numbers)   
    elif (type == 'Q'):
        return sum((1-pow(x[slot],2)) for x in numbers)/len(numbers)
    elif (type == 'R'):
        return sum((x[slot]*(x[slot]-1))/2 for x in numbers)/len(numbers)
    else:
        return 9999999999999999999999999

def calcPosterior(numbers,p,q,r):
    val = 1
    for i in range(9):
        val *=pow(p[i],(numbers[i]*(numbers[i]+1))/2)*pow(q[i],(1-pow(numbers[i],2)))*pow(r[i],(numbers[i]*(numbers[i]-1))/2)
    return val

def calcPrior(numbers):
    maxsize = len(numbers[1])+len(numbers[0])
    prior =  {'pos':len(numbers[1])/maxsize,'neg':len(numbers[0])/maxsize} 
    return prior

def toClassList(matrix):
    classList = []
    for line in matrix:
        classList.append([int(l) for l in line.split(',')])
    return classList

def loadData(filename):
    tttdata = open(filename,'r')
    listA = []
    listAtt = []
    for line in tttdata.readlines():
        listA += [line.split(',')]
    for line in listA:
        tempAmostra = {"atributos":[],"result":None}
        tempAmostra["atributos"] = [eval(str(line[:-1]).replace('b','-1').replace('x','1').replace('o','0'))]
        tempAmostra["result"] = line[-1].replace("positive","1").replace("negative","0")
        nvString = ""
        for i in tempAmostra["atributos"]:
            nvString+=str(i).replace('[','').replace(']','').replace("'",'')
        nvString += ","
        nvString += tempAmostra["result"]
        listAtt += nvString 
    return listAtt

def main():
    matrix = loadData('tic-tac-toe.data')
    with open('ttt_amostragem.txt', 'w', newline='') as txtfile:
        txtfile.writelines(matrix)
    matrix = open('ttt_amostragem.txt','r').readlines()
    cnjTreinamento,cnjTeste = splitDataset(matrix,0.67)
    allSamples = toClassList(matrix)
    #Para intervalo de confianca e pontual
    globalMLP = []
    globalSVM = []
    globalKNN = []
    globalSUM = []
    globalBAY = []

    # Friedman Table
    # [MLP],[SVM],[KNN],[BAYES],[SUM]
    setTestTable = []
    ######################################
    for i in range(10):
        mlpRes = []
        svmRes = []
        sumRes = []
        bayRes = []
        knnRes = []
        nOfFolds = 10
        dividedKSamples = subpart(nOfFolds,allSamples)
        for l in range(len(dividedKSamples)):
            lrning, testing = getFold(l,dividedKSamples)
            new = [i[:-1] for i in lrning]
            test = [[i[-1]] for i in lrning]
            #MLP
            net = nl.net.newff([[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]],[3,1])
            err = net.train(new,test,epochs=100,show=30,goal=0.05)
            fMLP = net.sim([i[:-1] for i in testing])
            resMLP = normalize(fMLP)
            #SVM
            clf = svm.SVC()
            clf.fit(new, [i[-1] for i in lrning])
            resSvm = clf.predict([i[:-1] for i in testing])
            #KNN
            kVizi = 7 #K-vizinhos IMPAR.
            resKNN, probPostKNN = getPredictionsKNN(lrning,testing,kVizi)
            #BAY
            trnSampleSep = separateByClass(lrning)
            Pneg = [calculateProbability('P',trnSampleSep[0],i) for i in range(9)]
            Qneg = [calculateProbability('Q',trnSampleSep[0],i) for i in range(9)]
            Rneg = [calculateProbability('R',trnSampleSep[0],i) for i in range(9)]
            Ppos = [calculateProbability('P',trnSampleSep[1],i) for i in range(9)]
            Qpos = [calculateProbability('Q',trnSampleSep[1],i) for i in range(9)]
            Rpos = [calculateProbability('R',trnSampleSep[1],i) for i in range(9)]
            prior = calcPrior(trnSampleSep) 
            summaries = [{'p':Ppos,'q':Qpos,'r':Rpos,'prior':prior['pos']},{'p':Pneg,'q':Qneg,'r':Rneg,'prior':prior['neg']}]
            bayPredictions,probPostB = getPredictions(summaries,testing)
            #SUM
            sumPredictions = []
            for i in range(len(testing)):
                if probPostB[i][0] + probPostKNN[i][1] > probPostB[i][1] + probPostKNN[i][0]:
                    sumPredictions.append(0)
                if probPostB[i][0] + probPostKNN[i][1] <= probPostB[i][1] + probPostKNN[i][0]:
                    sumPredictions.append(1)        
            ##############################
            resultMLP = getAccuracy(testing,resMLP)
            resultSVM = getAccuracy(testing,resSvm)
            resultSUM = getAccuracy(testing,sumPredictions)
            resultKNN = getAccuracy(testing,resKNN)
            resultBAY = getAccuracy(testing,bayPredictions)
            ##
            mlpRes.append(resultMLP)
            svmRes.append(resultSVM)
            sumRes.append(resultSUM)
            knnRes.append(resultKNN)
            bayRes.append(resultBAY)
            ranks = [resultMLP,resultSVM,resultKNN,resultSUM,resultBAY]
            setTestTable.append(friedmanRank(ranks))
            print('Result (MLP:{0} || SVM:{1} || SUM:{2} || KNN:{3} || BAY:{4})'.format(resultMLP,resultSVM,resultSUM,resultKNN,resultBAY))
        finResMLP = sum(mlpRes)/len(mlpRes)
        finResSVM = sum(svmRes)/len(svmRes)
        finResKNN = sum(knnRes)/len(knnRes)
        finResSUM = sum(sumRes)/len(sumRes)
        finResBAY = sum(bayRes)/len(bayRes)
        globalMLP+=mlpRes
        globalSVM+=svmRes
        globalKNN+=knnRes
        globalSUM+=sumRes
        globalBAY+=bayRes
    dotResMLP = sum(globalMLP)/len(globalMLP)
    dotResSVM = sum(globalSVM)/len(globalSVM)
    dotResKNN = sum(globalKNN)/len(globalKNN)
    dotResSUM = sum(globalSUM)/len(globalSUM)
    dotResBAY = sum(globalBAY)/len(globalBAY)
    meanRankMLP = sum([i[0] for i in setTestTable])/len(setTestTable)
    meanRankSVM = sum([i[1] for i in setTestTable])/len(setTestTable)
    meanRankKNN = sum([i[2] for i in setTestTable])/len(setTestTable)
    meanRankSUM = sum([i[3] for i in setTestTable])/len(setTestTable)
    meanRankBAY = sum([i[4] for i in setTestTable])/len(setTestTable)
    print('{0}'.format([meanRankMLP,meanRankSVM,meanRankKNN,meanRankSUM,meanRankBAY]))
    meanRanks = [meanRankMLP,meanRankSVM,meanRankKNN,meanRankSUM,meanRankBAY]
    Xf = testStatistic(100,5,meanRanks)

    friedResult = Ff(100,5,Xf)
    print('Xf: {0}'.format(Xf))
    #Teste F
    alpha = 0.05
    valor_critico = scipy.stats.f.cdf(1-alpha,5,100)
    if(friedResult>valor_critico):
        print('Rejeita H0, {0} > {1}, logo os classificadores tem medias diferentes.'.format(friedResult,valor_critico))
        #Rejeita hipotese nula. Os dois sao diferentes.
        #Nemenyi
    else:
        print('Aceita H0, {0} <= {1}, logo os classificadores tem medias iguais.'.format(friedResult,valor_critico))
        #Aceita hipotese nula. Os dois classificadores tem a mesma media.
    
    nem = nemenyiGET(5,100)
    for i in range(len(meanRanks)):
        for k in range(len(meanRanks[:i+1])):
            if (math.fabs(meanRanks[i] - meanRanks[k]) < nem) and (i!=k):
                print('{0} e {1} tem medias iguais (CD:{2} > {3} - {4})'.format(i,k,nem,meanRanks[i],meanRanks[k]))
            if (math.fabs(meanRanks[i] - meanRanks[k]) > nem) and (i!=k):
                print('{0} e {1} tem medias diferentes (CD:{2} > {3} - {4})'.format(i,k,nem,meanRanks[i],meanRanks[k]))
    ########FRIEDMAN########
    mResMLP,iMLP,eMLP = mean_confidence_interval(globalMLP)
    mResSVM,iSVM,eSVM = mean_confidence_interval(globalSVM)
    mResKNN,iKNN,eKNN = mean_confidence_interval(globalKNN)
    mResSUM,iSUM,eSUM = mean_confidence_interval(globalSUM)
    mResBAY,iBAY,eBAY = mean_confidence_interval(globalBAY)
    print('Estimativa pontual de MLP: {0}'.format(1-mResMLP/100))
    print('Estimativa pontual de SVM: {0}'.format(1-mResSVM/100))
    print('Estimativa pontual de KNN: {0}'.format(1-mResKNN/100))
    print('Estimativa pontual de SUM: {0}'.format(1-mResSUM/100))
    print('Estimativa pontual de BAY: {0}'.format(1-mResBAY/100))
    print('Intervalo de Conf. de MLP: {0} ~ {1} - 95%'.format(iMLP,eMLP))
    print('Intervalo de Conf. de SVM: {0} ~ {1} - 95%'.format(iSVM,eSVM))
    print('Intervalo de Conf. de KNN: {0} ~ {1} - 95%'.format(iKNN,eKNN))
    print('Intervalo de Conf. de SUM: {0} ~ {1} - 95%'.format(iSUM,eSUM))
    print('Intervalo de Conf. de BAY: {0} ~ {1} - 95%'.format(iBAY,eBAY))

def nemenyiGET(k,N):
    alpha = 2.728   #1-alpha (0.05), da tabela do artigo da pagina da cadeira
    return alpha*(math.sqrt((k*(k+1))/(6*N)))


def friedmanRank(k):
    ranking = []
    currRank = 0
    for i in range(len(k)):
        currRank = 1
        for l in range(len(k)):
            if k[i] < k[l]:
                currRank = currRank + 1
            elif (k[i] == k[l] and i != l):
                currRank = currRank + 0.5
        ranking.append(currRank)
    return ranking

def getFold(n,samples):
    learning = samples[:n]+samples[n+1:]
    learn = []
    for i in learning:
        for l in i:
            learn.append(l)
    test = samples[n]
    return learn, test

###########FRIEDMAN#############
#Computa as estatisticas de teste
def testStatistic(N,k,rankMeanTable):
    value1 = (12*N)/(k*(k+1))
    value2 = 0
    for i in rankMeanTable:
        value2 += pow(i,2)
    value3 = (k*pow((k+1),2)/4)
    return value1*(value2-value3)

def Ff(N,k,Xf):
    return ((N-1)*Xf)/(N*(k-1)-Xf)
###############################

#############################
def subpart(n,samples):
    trainSize = int(len(samples) * (1/n))
    propCls1 = estratif(samples)
    nCls1 = 0
    copy = samples
    trainSet = []
    for a in range(n):
        trainSet.append([])
    while len(copy) >= trainSize:
        for f in range(len(trainSet)):
            currStratif = estratif(trainSet[f])
            if currStratif > propCls1:
                index = 0
                while(copy[index][-1] == 1):
                    index = random.randrange(len(copy))
                trainSet[f].append(copy.pop(index))
            if currStratif <= propCls1:
                index = 0
                while(copy[index][-1] == 0):
                    index = random.randrange(len(copy))
                trainSet[f].append(copy.pop(index))               
    return trainSet


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t.ppf((1+confidence)/2., n - 1)
    return m, m-h, m+h

def estratif(samples):
    if len(samples)>0:
        pct = 0
        for i in samples:
            if i[-1] == 1:
                pct = pct +1
        return pct/len(samples)
    return 0

def normalize(rr):
    result = []
    for r in rr:
        if r[0] < 0.5:
            result.append(0)
        if r[0] >= 0.5:
            result.append(1)
    return result

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

###########################################################################
# ALGORITMO K NEAREST NEIGHBORS 

def getPosteriorKNN(probPost):
    postP = 0
    postN = 0
    for i in probPost:
        postP += i[0]
        postN += i[1]
    postP = postP/(len(probPost))
    postN = postN/(len(probPost))
    return postP, postN

def summarizeByClassKNN(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def delta(i,j):
    if i != j:
        return 1
    else:
        return 0

def dissimilarity(x1,x2):
    valor = sum([delta(x1[i],x2[i]) for i in range(len(x1)-1)])
    return valor
        
def calculate_dmatrix(matrix):
    matrix_heigth = len(matrix)
    matrix_width = len(matrix[0][:-1])
    posi_lines = []
    nega_lines = []
    dmatrix = []
    for i in range(matrix_heigth):
        new_line = []
        for j in range(matrix_heigth):
            new_line.append(dissimilarity(matrix[i],matrix[j]))
        dmatrix.append(new_line)
    return dmatrix

def calculateKNN(trnSamples,inputVector,k):
    dissimMatrix = []
    for i in trnSamples:
        dissimMatrix.append([dissimilarity(i,inputVector),i[-1]])
    dissimMatrix = sorted(dissimMatrix, key=lambda matrix:matrix[0])
    kVizinhosEscolhidos = []
    for i in range(0,(k-1)):
        kVizinhosEscolhidos.append(dissimMatrix[i])
    votosP = 0
    votosN = 0
    for i in kVizinhosEscolhidos:
        if i[-1] == 0:
            votosN = votosN + 1
        elif i[-1] == 1:
            votosP = votosP + 1
    votos = votosP - votosN
    posteriori = [(votosP/len(kVizinhosEscolhidos)),(votosN/len(kVizinhosEscolhidos))]
    return votos, posteriori

def predictKNN(trnSamples, inputVector,k):
    probabilities,posteriori = calculateKNN(trnSamples, inputVector,k)
    bestLabel = None
    if probabilities > 0:
        bestLabel = 1
    else:
        bestLabel = 0
    return bestLabel, posteriori

def getPredictionsKNN(trnSamples, testSet,k):
    predictions = []
    posteriori = []
    for i in range(len(testSet)):
        result,post = predictKNN(trnSamples, testSet[i],k)
        predictions.append(result)
        posteriori.append(post)
    return predictions,posteriori
###########################################################################

###########################################################################
# ALGORITMO BAYESIANO

def calcConditional(numbers,p,q,r):
    val = 1
    for i in range(9):
        val *=pow(p[i],(numbers[i]*(numbers[i]+1))/2)*pow(q[i],(1-pow(numbers[i],2)))*pow(r[i],(numbers[i]*(numbers[i]-1))/2)
    return val

def calcPrior(numbers):
    maxsize = len(numbers[1])+len(numbers[0])
    prior =  {'pos':len(numbers[1])/maxsize,'neg':len(numbers[0])/maxsize} 
    return prior

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateClassProbabilities(summaries,inputVector):
    varsum = 0
    for i in summaries:
        varsum += i['prior']*calcConditional(inputVector,i['p'],i['q'],i['r'])
    posiProb = (summaries[0]['prior']*calcConditional(inputVector,summaries[0]['p'],summaries[0]['q'],summaries[0]['r']))/varsum
    negProb = (summaries[1]['prior']*calcConditional(inputVector,summaries[1]['p'],summaries[1]['q'],summaries[1]['r']))/varsum
    return [negProb,posiProb]    

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    posteriorProb = []
    for i in range(len(probabilities)):
        posteriorProb.append(probabilities[i])
        if bestLabel is None or probabilities[i] > bestProb:
            bestProb = probabilities[i]
            bestLabel = i
    return bestLabel,posteriorProb

def getPredictions(summaries, testSet):
    predictions = []
    postProb = []
    for i in range(len(testSet)):
        result,posteriorProb = predict(summaries, testSet[i])
        predictions.append(result)
        postProb.append(posteriorProb)
    return predictions,postProb

###########################################################################

if __name__ == "__main__":
    main()
