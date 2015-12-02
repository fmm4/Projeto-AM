from Dissimilarity import Dissimilarity
from random import randint

class SFCMdd(object):
    def __init__(self, training_set, dissimilarity_matrix):
        self.__E = training_set
        self.__D = dissimilarity_matrix
        self.__K = 2
        self.__G = []
        self.__U = []
        self.__n = 0
        self.__m = 2
        self.__J = 0.0
        self.__q = 2
        self.d = Dissimilarity(training_set)

    def pick_prototypes(self):
        all_values = []
        while len(all_values) < (self.__K * self.__q):
            element = self.__E[randint(0,self.__n-1)]
            if element not in all_values:
                all_values.append(element)
        i = 0
        newG = []
        for j in range(self.__K):
            Gi = []
            while len(Gi) < self.__q:
                Gi.append(all_values[i])
                i+=1
            newG.append(Gi)
        return newG

    def membership_degree(self,element):
        ui = []
        t1=0
        t2=0
        for ek in self.__G:
            uik = []
            t1 = sum([self.d.dissimilarity(element,e)+1 for e in ek])
            for eh in self.__G:
                values = [self.d.dissimilarity(element,e)+1 for e in eh]
                t2 = sum(values)
                uik.append((t1/t2)**(1/(self.__m-1)))
            ui.append(sum(uik)**(-1))

        return ui

    def adequacy_criterion(self):
        j_values = []
        for k in range(self.__K):
            n_values = []
            for i in range(self.__n):
                ui = self.__U[i]
                uik = ui[k]
                ei = self.__E[i]
                sum_d = sum([self.d.dissimilarity(ei,e) for e in self.__G[k]])
                n_values.append((uik**(self.__m))*sum_d)
            j_values.append(sum(n_values))

        return sum(j_values)

    def step1(self):
        newG = []
        for k in range(self.__K):
            l = []
            l_values = []
            for eh in self.__E:
                l_value = 0.0
                l_values_of_h = []
                for i in range(self.__n):
                    ei = self.__E[i]
                    ui = self.__U[i]
                    l_values_of_h.append((ui[k]**self.__m)*self.d.dissimilarity(ei,eh))
                l_values.append([sum(l_values_of_h),eh])

            l_values.sort(key=lambda tup: tup[0])
            el = [eh for sum_l,eh in l_values]
            newG.append(el[:self.__K])
        return newG

    def compute(self, K=2, T=150, emax=(10.e-10), m=2, q=2):
        # Initialization
        error = 1.0
        t = 0
        self.__n = len(self.__E)
        self.__K = K
        self.__q = q
        # Randomly select K distinct prototypes Gk
        self.__G = self.pick_prototypes()
        # For each object ei compute its membership degree uik

        self.__U = [self.membership_degree(self.__E[i]) for i in range(self.__n)]
        self.__J = self.adequacy_criterion()

        while error > emax and t < T:
            # Computation of the Best Prototypes
            t = t + 1
            #print("U: "+str(self.__U[:5]))
            #print("G: "+str(self.__G))
            self.__G = self.step1()
            # Definition of the Best Fuzzy Partition
            self.__U = [self.membership_degree(element) for element in self.__E]
            # Stopping Criterion
            J_t = self.adequacy_criterion()
            error = abs(J_t - self.__J)
            self.__J = J_t
            print("Iteration "+str(t)+"...")

        if error < emax:
            print("Stopped with error: "+str(error))
        elif t >= T:
            print("Stopped with "+str(t)+" Iterations")

        return [self.__U,self.__G,self.__J]
