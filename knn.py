import sys
import numpy as np
import math
from collections import Counter
#strip labels from input data return set with labels
def getTrainStructure(infile):
    f = open(infile)
    M,N =[int(x) for x in next(f).split()]
    #get point values
    Narray = [[[]for i in range(N+1)] for i in range(M)]
    Labels = [[] for i in range(M)]
    for x in range(M) :
        Narray[x]=[float(y) for y in next(f).split()]
    Narray = np.matrix(Narray)
    Labels = Narray[:,N]
    Clean = Narray[:,0:N]
    
    return Clean,Labels

#return test data set
def getTestStructure(infile):
    f = open(infile)
    M,N =[int(x) for x in next(f).split()]
    #get point values
    Narray = [[[]for i in range(N)] for i in range(M)]
    for x in range(M) :
        Narray[x]=[float(y) for y in next(f).split()]
    Narray = np.matrix(Narray)
    return Narray

#find distance from one point in T to all of P and store in Dist array
def pointDistances(Tpoint,P):
    Dist = list()
    for i in range(0,len(P)):
        Dist.append(np.linalg.norm(Tpoint-P[i]))
    return Dist

#create ranking order array to give indeces of smallest distances to a point
def rankArray(D):
    return np.argsort(D)

#find k nearest neighbors from the ranked array (which indexes the label array and picks majority)
def knn(K, R, L):
    Class = R[0:K].copy()
    for i in range(0,K):
        #R[i] holds index of nearest neighbor
        #L holds label of nn
        Class[i] = (L[R[i]])
    return Class

#get majority from list of nearest neighbor class
def majLabel(C):
    #create map to check occurence of unique labels
    Count=[]
    #check k nearest neighbors
    for i in range(len(C)):
        #helper to see if label is already in Count[]
        inC = False
        #loop through map of labels, check if encounter another duplicate and increment Count[i][j++]
        for j in range(len(Count)):
            #if Count has already encountered the current label, add it to its respective holder
            if(Count[j][0]==C[i]):
                inC = True
                Count[j] = (Count[j][0],Count[j][1]+1)
        #label is not in Count[], so add a tuple and start with value 1
        if(inC == False):
            Count.append((C[i],1))
    #find max from second part of tuple in counts, return the Value!
    Label = max(Count, key=lambda x:x[1])
    return Label[0]

k = int(sys.argv[1])
P,L = getTrainStructure(sys.argv[2])
T = getTestStructure(sys.argv[3])
#P is list of training points with L their corresponding labels
#we need to compare the k closest points for every point in T to P 
#then assign majority of L to label for T
#TL will hold the final decision for each value at T
TL = list()

for i in range(len(T)):
    #D get distances from all points to T[i]
    D = pointDistances(T[i],P)
    #Ra rank distances from shortest to longest
    Ra = rankArray(D)
    #Class get classes from k shortest distances in Ra
    Class = knn(k,Ra,L)
    #Label picks majority from Class
    Label = majLabel(Class)
    TL.append(Label)

#print out test data with labels
def printIt(T,TL):
    for i in range(len(T)):
        
        print str(i+1),". ",str(T[i]).replace('[','').replace(']','')," -- ",str(TL[i])


printIt(T,TL)

