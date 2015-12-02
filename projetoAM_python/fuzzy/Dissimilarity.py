class Dissimilarity(object):
    def __init__(self,matrix):
        self.__matrix = matrix

    def delta(self,i,j):
        if i == j:
            return 1
        else:
            return 0

    def dissimilarity(self,x1,x2):
        return sum([self.delta(x1[i],x2[i]) for i in range(len(x1))])

    def calculate_dmatrix(self):
        matrix_heigth = len(self.__matrix)
        matrix_width = len(self.__matrix[0][:-1])

        dmatrix = []
        for i in range(matrix_heigth):
            new_line = []
            for j in range(matrix_heigth):
                new_line.append(self.dissimilarity(self.__matrix[i],self.__matrix[j]))
            dmatrix.append(new_line)

        return dmatrix
