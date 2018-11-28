# we build a list of lists, which contains the matrices sotred accordning to their classIndex

import numpy as np
import math
import sys

filepath=sys.argv[1]

numbOfClasses = 0
arraySize = 0
obSize = 0

meansByClass = []
covarsByClass = []
priorsByClass = []


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
	listOfLists = [ [] for i in range(numbOfClasses+1) ]
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
	priors = [ 0 for i in range(numbOfClasses+1) ]
	for x in range(numbOfClasses+1) :
		priors[x%10] = np.divide(len(listOfLists[x%10]),obSize)
	return priors

def inverseCovars_full(covars):
	invCov = []
	for sig in covars:
		invCov.append(np.linalg.inv(sig))
	return invCov

def invsersCovars_diag(covars):
	invCovs = []
	for diagVector in covars:
		invCovs.append(np.reciprocal(diagVector))
	return invCovs

def determinants(covars):
	determinants = []
	for covar in covars:
		determinants.append(np.prod(covar))
	return determinants

def logConstTerm(determinants):
	global arraySize
	lCTerms = []
	for det in determinants:
		num = (-1/2)*(arraySize*np.log(2*math.pi)+np.log(det))
		lCTerms.append(num)
	return lCTerms

def logPriors(priors):
	logPrios =[]
	for prior in priors:
		logPrios.append(np.log(prior))
	return logPrios

def quadratic_diag(mean,invCovar,x):
	vec = x - mean
	return np.sum(np.multiply(invCovar,vec**2))

def quadratic_full(mean,invCovar,x):
	vec = x- mean
	temp = invCovar*vec
	return np.dot(vec,temp)


def decisionRule_full(logPriors,invCovs,means,logConstTerms,x):
	global numbOfClasses
	maxIndex = -1
	max = 0
	for k in range(numbOfClasses+1):
		disc = logPriors[k%10] + logConstTerms[k%10] + quadratic_diag(means[k%10],invCovs[k%10],x)
		if disc >= max:
			max = disc
			maxIndex = k
	return maxIndex

def decisionRule_diag(logPriors,invCovs,means,logConstTerms,x):
	global numbOfClasses
	maxIndex = -1
	max = 0
	for k in range(numbOfClasses+1):
		disc = logPriors[k%10] + logConstTerms[k%10] + quadratic_diag(means[k%10],invCovs[k%10],x)
		if disc >= max:
			max = disc
			maxIndex = k
	return maxIndex


def train():
	global numbOfClasses
	global arraySize
	global filepath
	global meansByClass
	global priorsByClass
	global covarsByClass

	file = open(filepath)
	readHeader(file)

	observationsByClass = sortByClass(file)
	meansByClass = calculateMeans(observationsByClass)
	priorsByClass = calPrior(observationsByClass)
	covarsByClass = sigmaSquared(observationsByClass,meansByClass)


def test(filename):
	global meansByClass
	global priorsByClass
	global covarsByClass

	lPriors = logPriors(priorsByClass)
	logConstTerms = logConstTerm(determinants(covarsByClass))
	invCovs = invsersCovars_diag(covarsByClass)

	mistakesMatrix = np.zeros((10,10),dtype = int)

	file = open(filename)
	readHeader(file)
	testInstancesByClass = sortByClass(file)
	numberOfTests = 0
	numberOfMistakes = 0
	for k in range(11):
		numberOfTests = numberOfTests + len(testInstancesByClass[k%10])
		for observation in testInstancesByClass[k%10]:
			foundClass = decisionRule_diag(lPriors,invCovs,meansByClass,logConstTerms,observation)
			if foundClass != k:
				numberOfMistakes = numberOfMistakes + 1
				mistakesMatrix[k%10,foundClass%10] = mistakesMatrix[k%10,foundClass%10] + 1

	errorRate =numberOfMistakes / numberOfTests
	print(errorRate)
	print(mistakesMatrix)


def run():
	train()
	test("usps.test")






run()
