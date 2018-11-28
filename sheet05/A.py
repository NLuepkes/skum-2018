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
		listOfM.append(np.divide(sum, len(listOfLists[index])))
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
	priors = [ 0 for i in range(numbOfClasses) ]
	for x in range(numbOfClasses) :
		priors[x%10] = np.divide(len(listOfLists[x%10]),obSize)
	return priors


def inverseCovars_full(covars):
	# list of matrices
	invCov = []
	for sig in covars:
		invCov.append(np.linalg.inv(sig))
	return invCov

def invsersCovars_diag(covars):
	# list of vectors
	invCovs = []
	for diagVector in covars:
		invCovs.append(np.reciprocal(diagVector))
	return invCovs

def determinants_diag(covars):
	# nur für Diagmatritzen
	determinants = []
	for covar in covars:
		determinants.append(np.prod(covar))
	return determinants

def logConstTerm(determinants):
	global arraySize
	lCTerms = []
	for det in determinants:
		num = (-1/2)*(arraySize*(np.log(2) + np.log(math.pi))+np.log(det))
		lCTerms.append(num)
	return lCTerms

def logPriors(priors):
	logPrios = []
	for prior in priors:
		logPrios.append(np.log(prior))
	return logPrios

def quadratic_diag(mean,invCovar,x):
	vec = x - mean
	return (-1/2)*np.sum(np.multiply(invCovar,vec**2))

def quadratic_full(mean,invCovar,x):
	vec = x - mean
	temp = invCovar*vec
	return (-1/2)*np.dot(vec,temp)


def decisionRule_full(logPriors,invCovs,means,logConstTerms,x):
	global numbOfClasses
	maxIndex = -1
	max = -sys.maxsize
	for k in range(numbOfClasses):
		disc = logPriors[k] + logConstTerms[k] + quadratic_full(means[k],invCovs[k],x)
		if disc >= max:
			max = disc
			maxIndex = k
	return maxIndex

def decisionRule_diag(logPriors,invCovs,means,logConstTerms,x):
	global numbOfClasses
	maxIndex = -1
	max = -sys.maxsize
	probs = []
	for k in range(numbOfClasses):
		disc = logPriors[k] + logConstTerms[k] + quadratic_diag(means[k],invCovs[k],x)
		probs.append(disc)
		if disc >= max:
			max = disc
			maxIndex = k
	if maxIndex == -1 :
		print("maxIndex == -1 => kein Max gefunden")
	#print(probs)
	#exit(0)
	return maxIndex




##### SHEET 05 ####

# a)

# class specific full covariance matrix
def csfullcovm(listOfLists, listMeans) :
	listOfSig = []
	for index in range(numbOfClasses):
		sum = np.zeros((256,256), dtype = float)
		for observation in listOfLists[index % numbOfClasses]:
			x = observation - listMeans[index]
			x = np.outer(x, x) # x*x^T
			sum = np.add(sum, x)
		listOfSig.append(np.divide(sum, len(listOfLists[index])))
	return listOfSig

def csdiagcovm(listOfLists, listMeans) :
	listOfSig = []
	for index in range(numbOfClasses):
		sum = np.zeros(256, dtype = float)
		for observation in listOfLists[index % numbOfClasses]:
			x = observation - listMeans[index]
			x = np.square(x)
			sum = np.add(sum, x)
		listOfSig.append(np.divide(sum, len(listOfLists[index])))
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
	covarsByClass = [covarsByClass for n in range(0,10)]
	
	covarsByClass = csdiagcovm(observationsByClass, meansByClass)


def test(filename):
	global meansByClass
	global priorsByClass
	global covarsByClass

	lPriors = logPriors(priorsByClass)
	logConstTerms = logConstTerm(determinants_diag(covarsByClass))
	logConstTerms = [0]*10
	invCovs = invsersCovars_diag(covarsByClass)

	mistakesMatrix = np.zeros((10,10),dtype = int)

	file = open(filename)
	readHeader(file)
	testInstancesByClass = sortByClass(file)
	numberOfTests = 0
	numberOfMistakes = 0
	for k in range(10):
		#print('regocniing all of class {}'.format(k))
		numberOfTests = numberOfTests + len(testInstancesByClass[k%10])
		for observation in testInstancesByClass[k%10]:
			# für Blatt 03 ist _diag ok, ansich aber allgemein machen
			foundClass = decisionRule_diag(lPriors,invCovs,meansByClass,logConstTerms,observation)
			if foundClass != k:
				numberOfMistakes = numberOfMistakes + 1
				mistakesMatrix[k%10,foundClass%10] = mistakesMatrix[k%10,foundClass%10] + 1

	errorRate = numberOfMistakes / numberOfTests
	print(errorRate)
	print(mistakesMatrix)


def run():
	train()
	test("usps.test")
	
run()
