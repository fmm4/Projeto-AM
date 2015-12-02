import csv
from Dissimilarity import Dissimilarity
from SFCMdd import SFCMdd
from sklearn.metrics.cluster import adjusted_rand_score

def loadData(filename):
    tttdata = open(filename,'r')
    matrix = []
    for line in tttdata.readlines():
        matrix += [line.split(',')]
    return matrix

def hard_partition_generator(fuzzy_partition):
    for ui in fuzzy_partition:
        if(ui[0] > ui[1]):
            yield [1,0]
        else:
            yield [0,1]

def write_csv_partition(partition,filename):
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["G1","G2"])
        for line in partition:
            spamwriter.writerow(line)

def main():
    ttt_data = loadData('tic-tac-toe.data')
    training_set = [row[:9] for row in ttt_data]
    #print(training_set)
    dmatrix = Dissimilarity(training_set).calculate_dmatrix()

#    with open('ttt_dmatrix.csv', 'w', newline='') as csvfile:
#        spamwriter = csv.writer(csvfile,
#                                delimiter=',',
#                                quotechar='|',
#                                quoting=csv.QUOTE_MINIMAL)
#
#        spamwriter.writerow(['x'+str(i) for i in range(len(dmatrix))])
#        for line in dmatrix:
#            spamwriter.writerow(line)

    results = []
    sfcmdd = SFCMdd(training_set,dmatrix)
    for i in range(100):
        U,G,J = sfcmdd.compute(K=2,T=150,emax=(10.e-10),m=2,q=2)
        success=0
        fail=0
        for y,n in U[:626]:
            if y < n: fail+=1
            else: success+=1
        for y,n in U[626:]:
            if n < y: fail+=1
            else: success+=1
        #print("RESULTS: \n>>>>> sucess: "+str(success)+"\n>>>>> fail: "+str(fail))
        print("Classification Rate: "+str(success/958.0))
        results.append([J,U,(success/958.0)])

    results.sort(key=lambda tup: tup[0])
    for i in results[:10]:
        print("J: "+results[0])
        print("Rate: "+results[2])

    fuzzy_partition, prototypes, best_rate = results[0]
    hard_partition = [e for e in hard_partition_generator(fuzzy_partition)]
    ars = adjusted_rand_score(
        [e1 for e1,e2 in hard_partition],
        [e2 for e1,e2 in hard_partition]
    )

    write_csv_partition(fuzzy_partition, "fuzzy_k_medoids_result.csv")
    write_csv_partition(hard_partition, "hard_partition.csv")
    write_csv_partition([ars], "adjusted_rand_score.csv")

if __name__ == "__main__":
    main()
