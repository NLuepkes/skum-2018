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

# part b
def exb():
	global numbOfClasses
	global arraySize

	file = open(filepath)
	readHeader(file)

	listOfLists = sortByClass(file)
	means = calculateMeans(listOfLists)
	sigsq = sigmaSquared(listOfLists, means)
	prior = calPrior(listOfLists)

	file = open("usps_d.param", "w")
	file.write("d"+"\n")
	file.write(str(numbOfClasses)+"\n")
	file.write(str(arraySize)+"\n")

	for x in range(1,numbOfClasses+1) :
		file.write(str(x)+"\n")
		file.write(str(prior[x%numbOfClasses]))
		file.write("\n")
		for i in means[x%numbOfClasses] :
			file.write(str(i)+" ")
		file.write("\n")
		for i in sigsq :
			file.write(str(i))
		file.write("\n")
	return print("done!")

exb()
