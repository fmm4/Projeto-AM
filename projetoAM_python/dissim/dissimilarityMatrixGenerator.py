import csv
import time

def loadData(filename):
    tttdata = open(filename,'r')
    matrix = []
    for line in tttdata.readlines():
        matrix += [line.split(',')]
    return matrix

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

    dmatrix = []
    for i in range(matrix_heigth):
        new_line = []
        for j in range(matrix_heigth):
            new_line.append(dissimilarity(matrix[i],matrix[j]))
        dmatrix.append(new_line)

    return dmatrix

def main():
    matrix = loadData('tic-tac-toe.data')
    
    dmatrix = calculate_dmatrix(matrix)
    with open('ttt_dmatrix.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['x'+str(i) for i in range(len(dmatrix))])
        for line in dmatrix:
            spamwriter.writerow(line)
    

if __name__ == "__main__":
    main()
    time.sleep(2)
