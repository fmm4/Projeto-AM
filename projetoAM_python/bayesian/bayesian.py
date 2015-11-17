import csv
import time
import random
import math 

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

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
    classList = []
    for line in matrix:
        classList.append([int(l) for l in line.split(',')])
    fileWriteTTT = open('ttt_amostragem.txt','w')
    summarizedByClass =  summarizeByClass(cnjTreinamento)
    fileWriteTTT.write(('SummarizedByClass: {0}').format(summarizedByClass))
    inputVector = [1,-1,-1,0,1,0,-1,-1,1]
    probabilities = calculateClassProbabilities(summarizedByClass, inputVector)
    print("Pertence a {0}".format(probabilities))
    time.sleep(99)

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

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
