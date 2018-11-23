# we build a list of lists, which contains the matrices sotred accordning to their classIndex

import numpy as np
import sys

filepath=sys.argv[1]

numbOfClasses = 0
arraySize = 0
obSize = 0

# part a

def readIn(file):
	readHeader(file)
	sortByClass(file)

def readHeader(file):
	global numbOfClasses
	global arraySize
	numbOfClasses = int(file.readline())
	arraySize = int(file.readline())

def sortByClass(file):
	global numbOfClasses
	global obSize
	listOfLists = [ [] for i in range(numbOfClasses) ]
	flag = True
	while flag :
		flag = readMatr(file, listOfLists)
	for x in listOfLists:
		obSize += len(x)
	return listOfLists

# this function reads the classindex and appends the respective matrix to the list
# of all matrices with this classIndex
def readMatr(file, listOfClArr):
	try :
		line = file.readline()
		if( line == "") :
			return False
		classIndex = int(line.strip("\n"))
		arrAsLists = []
		for x in range(16) :
			arrAsLists.extend(file.readline().strip("\n").split())
		listOfClArr[classIndex%int(numbOfClasses)].append(np.asarray(arrAsLists, dtype = float))
		return True
	except EOFError :
		return False



def calculateMeans(listOfLists):
	listOfM = []
	for index in range(numbOfClasses):
		sum = np.zeros(256, dtype = float)
		for observation in listOfLists[index % numbOfClasses]:
			sum = np.add(sum, observation)
		listOfM.extend(np.divide(sum, len(listOfLists[index])))
	return listOfM

def sigmaSquared(listOfLists, means):
	global numbOfClasses
	global obSize
	v = np.zeros(256, dtype = float)
	for x in range(numbOfClasses) :
		for arr in listOfLists[x%numbOfClasses] :
			v = np.add(v, np.square( np.subtract(arr, means[x%numbOfClasses])))
	return np.divide(v, obSize).tolist()

def calPrior(listOfLists):
	global numbOfClasses
	global obSize
	priors = [0]*numbOfClasses
	for x in range(numbOfClasses) :
		priors[x] = np.divide(len(listOfLists[x]),obSize)
	return priors


##### SHEET 05 ####

# a)

# class specific full covariance matrix
def csfullcovm(listOfLists, listMeans) :
	listOfSig = []
	for index in range(numbOfClasses):
		sum = np.zeros((256,256), dtype = float)
		for observation in listOfLists[index % numbOfClasses]:
			x = observation - listMeans[index]
			x = np.matmul(x, np.matrix.transpose(x))
			sum = np.add(sum, x)
		listOfSig.extend(np.divide(sum, len(listOfLists[index])))
	return listOfSig
						  
def csdiagcovm(listOfLists, listMeans) : 
	listOfSig = []
	for index in range(numbOfClasses):
		sum = np.zeros(256, dtype = float)
		for observation in listOfLists[index % numbOfClasses]:
			x = observation - listMeans[index]
			x = np.square(x)
			sum = np.add(sum, x)
		listOfM.extend(np.divide(sum, len(listOfLists[index])))
	return listOfSig

def pfullcovm(listOfLists, listMeans) :
	sum = np.zeros((256,256), dtype = float)
	for index in range(numbOfClasses):
		for observation in listOfLists[index % numbOfClasses]:
			x = observation - listMeans[index]
			x = np.matmul(x, np.matrix.transpose(x))
			sum = np.add(sum, x)
	sum = np.divide(sum, obSize)
	return sum

def pdiagcovm(listOfLists, listMeans) : 
	sum = np.zeros(256, dtype = float)
	for index in range(numbOfClasses):
		for observation in listOfLists[index % numbOfClasses]:
			x = observation - listMeans[index]
			x = np.square(x)
			sum = np.add(sum, x)
	sum = np.divide(sum, obSize)
	return sum


def smooth(listOfLists, lam, means) :
	pooledM = pfullcovm(listOfLists, means)
	csM = cdfullcovm(listOfLists, means)
	sig = []
	for index in range(numbOfClasses) :
		sig.extend(lam*pooledM + (1-lam)*csM[index%numbOfClasses])
	return sig
		






















