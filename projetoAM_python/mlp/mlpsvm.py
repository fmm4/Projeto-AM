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
        return sum((x[slot]*x[slot]+1)/2 for x in numbers)/len(numbers)   
    elif (type == 'Q'):
        return sum((1-pow(x[slot],2)) for x in numbers)/len(numbers)
    elif (type == 'R'):
        return sum((x[slot]*x[slot]-1)/2 for x in numbers)/len(numbers)

def calcPosterior(numbers,p,q,r):
    val = 1
    for i in range(0,9):
        val *=pow(p[i],(numbers[i]*(numbers[i]+1))/2)*pow(q[i],(1-pow(numbers[i],2)))*pow(r[i],numbers[i]*(numbers[i]-1)/2)
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
    nOfFolds = 10
    dividedKSamples = subpart(nOfFolds,allSamples)
    # a,b,c = mean_confidence_interval([1.0,2.0,3.0,4.0,5.0,6.0])
    mlpRes = []
    svmRes = []
    # Friedman Table
    # Primeiro resultado associado ao teste de MLP, segundo ao de SVM
    setTestTable = []
    #########FRIEDMAN###########
    for l in range(len(dividedKSamples)):
        lrning, testing = getFold(l,dividedKSamples)
        new = [i[:-1] for i in lrning]
        test = [[i[-1]] for i in lrning]
        print(estratif(lrning))
        # MLP
        net = nl.net.newff([[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]],[6,1])
        err = net.train(new,test,epochs=500,show=30,goal=0.05)
        fMLP = net.sim([i[:-1] for i in testing])
        resMLP = normalize(fMLP)
        # SVM
        clf = svm.SVC()
        clf.fit(new, [i[-1] for i in lrning])
        resSvm = clf.predict([i[:-1] for i in testing])
        resultMLP = getAccuracy(testing,resMLP)
        resultSVM = getAccuracy(testing,resSvm)
        mlpRes.append(resultMLP)
        svmRes.append(resultSVM)
        #########FRIEDMAN#########
        if(resultMLP==resultSVM):
            setTestTable.append([1.5,1.5])
        elif(resultMLP>resultSVM):
            setTestTable.append([1,2])
        else:
            setTestTable.append([2,1])
        ##########FRIEDMAN##########
        print('Teste (MLP:{0} ||'.format(mlpRes[-1])+'SVM:{0})'.format(svmRes[-1]))
    finResMLP = sum(mlpRes)/len(mlpRes)
    finResSVM = sum(svmRes)/len(svmRes)
    icResMLP,what,isd = mean_confidence_interval(mlpRes)
    icResSVM,whate,isde = mean_confidence_interval(svmRes)
    ########FRIEDMAN########
    meanRankMLP = sum([i[0] for i in setTestTable])/len(setTestTable)
    meanRankSVM = sum([i[1] for i in setTestTable])/len(setTestTable)
    print('Media dos ranks MLP: {0} de {1}'.format(meanRankMLP,[i[0] for i in setTestTable]))
    print('Media dos ranks SVM: {0} de {1}'.format(meanRankSVM,[i[1] for i in setTestTable]))
    Xf = testStatistic(10,2,[meanRankMLP,meanRankSVM])
    friedResult = Ff(10,2,Xf)
    print('Xf: {0}'.format(Xf))
    #Teste F
    alpha = 0.05
    valor_critico = scipy.stats.f.cdf(1-alpha,1,9)
    if(friedResult>valor_critico):
        print('rej, {0} > {1}'.format(friedResult,valor_critico))
        #Rejeita hipotese nula. Os dois sao diferentes.
        #Nemenyi
    else:
        print('rej, {0} <= {1}'.format(friedResult,valor_critico))
        #Aceita hipotese nula. Os dois classificadores tem a mesma media.
    ########FRIEDMAN########
    print('Estimativa pontual de MLP: {0}'.format(1-finResMLP/100))
    print('Estimativa pontual de SVM: {0}'.format(1-finResSVM/100))
    print('Intervalo de Conf. de MLP: {0} e {1}'.format(what,isd))
    print('Intervalo de Conf. de SVM: {0} e {1}'.format(whate,isde))

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
        value2 = value2 + (i - (k*pow((k+1),2)/4))
    return value1*value2 

def Ff(N,k,Xf):
    return ((N-1)*Xf)/(N*(k-1)-Xf)
####################

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

def getPredictions(trnSamples, testSet,k):
    predictions = []
    posteriori = []
    for i in range(len(testSet)):
        result,post = predict(trnSamples, testSet[i],k)
        predictions.append(result)
        posteriori.append(post)
    return predictions,posteriori

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

if __name__ == "__main__":
    main()
