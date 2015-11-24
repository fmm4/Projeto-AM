import csv
import time
import random
import math 
import itertools


def calculateProbability(type,numbers,slot):
    if (type == 'P'):
        # exponent = math.exp((x*(x+1))/2)
        return sum((x[slot]*x[slot]+1)/2 for x in numbers)/len(numbers)   
    elif (type == 'Q'):
        # exponent = math.exp((x*(x+1))/2)
        return sum((1-pow(x[slot],2)) for x in numbers)/len(numbers)
    elif (type == 'R'):
        # exponent = math.exp((x*(x+1))/2)
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
    print("Loaded data.")
    matrix = open('ttt_amostragem.txt','r').readlines()
    cnjTreinamento,cnjTeste = splitDataset(matrix,0.67)
    allSamples = toClassList(matrix)
    trnSamples = toClassList(cnjTreinamento)
    tstSamples = toClassList(cnjTeste)
    print(('Dividindo {0} amostras em um conjunto de treinamneto com {1} amostras e um conjunto de teste com {2} amostras').format(len(matrix), len(cnjTreinamento), len(cnjTeste)))
    trnSampleSep = separateByClass(trnSamples)
    Pneg = [calculateProbability('P',trnSampleSep[0],i) for i in range(0,9)]
    Qneg = [calculateProbability('Q',trnSampleSep[0],i) for i in range(0,9)]
    Rneg = [calculateProbability('R',trnSampleSep[0],i) for i in range(0,9)]
    Ppos = [calculateProbability('P',trnSampleSep[1],i) for i in range(0,9)]
    Qpos = [calculateProbability('Q',trnSampleSep[1],i) for i in range(0,9)]
    Rpos = [calculateProbability('R',trnSampleSep[1],i) for i in range(0,9)]
    prior = calcPrior(trnSampleSep)
    with open('respostas_para_relatorio.txt', 'w', newline='') as txtfile:
        txtfile.writelines('Ppos:{0}'.format(Ppos))
        txtfile.writelines('Qpos:{0}'.format(Qpos))
        txtfile.writelines('Rpos:{0}'.format(Rpos))
        txtfile.writelines('Pneg:{0}'.format(Pneg))
        txtfile.writelines('Qneg:{0}'.format(Qneg))
        txtfile.writelines('Rneg:{0}'.format(Rneg))
        txtfile.writelines('Priori Pos:{0}'.format(prior['pos']))
        txtfile.writelines('Priori Neg:{0}'.format(prior['neg']))    
    summaries = [{'p':Ppos,'q':Qpos,'r':Rpos,'prior':prior['pos']},{'p':Pneg,'q':Qneg,'r':Rneg,'prior':prior['neg']}]
    result = getPredictions(summaries,tstSamples)
    accuracy = getAccuracy(tstSamples,result)
    print('Acruacia: {0}'.format(accuracy))

    time.sleep(99)

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

def calculateClassProbabilities(summaries,inputVector):
    varsum = 0
    for i in summaries:
        varsum += i['prior']*calcPosterior(inputVector,i['p'],i['q'],i['r'])
    posiProb = summaries[0]['prior']*calcPosterior(inputVector,summaries[0]['p'],summaries[0]['q'],summaries[0]['r'])/varsum
    negProb = summaries[1]['prior']*calcPosterior(inputVector,summaries[1]['p'],summaries[1]['q'],summaries[1]['r'])/varsum
    return {1:posiProb,0:negProb}    

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

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

if __name__ == "__main__":
    main()
