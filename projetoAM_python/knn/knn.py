import csv
import time
import random
import math 
import itertools

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
    trnSamples = toClassList(cnjTreinamento)
    tstSamples = toClassList(cnjTeste)
    trnSampleSep = separateByClass(trnSamples)
    kVizi = 5 #K-vizinhos IMPAR.
    predictions,probPost = getPredictions(trnSamples,tstSamples,kVizi)
    acruacia = getAccuracy(tstSamples,predictions)
    print('{0}'.format(acruacia))
    postP = 0
    postN = 0
    for i in probPost:
        postP += i[0]
        postN += i[1]
    postP = postP/(len(probPost))
    postN = postN/(len(probPost))
    with open('ttt_debug.txt','w',newline='') as txtfile:
        txtfile.writelines('Post(P):{0} Post(N): {1}'.format(postP,postN))


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



def predict(trnSamples, inputVector,k):
    probabilities,posteriori = calculateKNN(trnSamples, inputVector,k)
    bestLabel = None
    if probabilities > 0:
        bestLabel = 1
    else:
        bestLabel = 0
    return bestLabel, posteriori

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
