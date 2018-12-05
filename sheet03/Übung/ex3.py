# Exercise Sheet 3
# 1. Training of an Image Recognition System
# Annika Sachte & Miriam Ganz

import sys
import numpy

def read_data(datafile):
    instance = {}
    number_of_instances = 0
    data = open(datafile,'r')
    line_index = 1
    for line in data:
        l = line.split()
        if line_index == 1:
            number_of_classes = int(l[0])
        elif line_index == 2:
            number_of_dimensions = int(l[0])
        else:
            if len(l) == 1:
                k = int(l[0])
                x = {}
                number_of_instances += 1
                component = 1
                component_block = 1 # of 16
            elif component_block <= 16:
                for element in l:
                    x[component] = int(element)
                    component += 1
                component_block += 1
            instance[number_of_instances] = [k,x]
        line_index += 1
    data.close()
    return(number_of_classes,number_of_dimensions,instance)

# a)
def calculate_mean(number_of_classes,number_of_dimensions,instance):
    mean = {}
    N = {}
    for k in range(1,number_of_classes+1,1):
        N[k] = []
        for i in instance:
            if instance[i][0] == k:
                N[k].append(i)
        x = {}
        for component in range(1,number_of_dimensions+1,1):
            x[component] = 0
            for n in N[k]:
                x[component] += instance[n][1][component]
            x[component] *= (1/len(N[k]))
        mean[k] = x
    return(mean)

def calculate_variance(number_of_dimensions,instance,mean):
    variance2 = {}
    for d in range(1,number_of_dimensions+1,1):
        result = 0
        for n in instance:
            result += (instance[n][1][d] - mean[instance[n][0]][d]) ** 2
        result /= len(instance)
        variance2[d] = result
    return(variance2)

def calculate_covariance(number_of_dimensions,variance2):
    covariance = {}
    for d1 in range(1,number_of_dimensions+1,1):
        for d2 in range(1,number_of_dimensions+1,1):
            if d1 == d2:
                covariance[(d1,d2)] = variance[d1]
            else:
                covariance[(d1,d2)] = 0
    return(covariance)

def calculate_p(number_of_classes,instance):
    p = {}
    N = {}
    for k in range(1,number_of_classes+1,1):
        N[k] = []
        for i in instance:
            if instance[i][0] == k:
                N[k].append(i)
        p[k] = len(N[k])/len(instance)
    return(p)

# b) calculate mean, variance and pooled covariance
inst = read_data('usps.train')
number_of_classes = inst[0]
number_of_dimensions = inst[1]
instance = inst[2]
mean = calculate_mean(number_of_classes,number_of_dimensions,instance)
variance = calculate_variance(number_of_dimensions,instance,mean)
covariance = calculate_covariance(number_of_dimensions,variance)
prior = calculate_p(number_of_classes,instance)
N = {}
for k in range(1,number_of_classes+1,1):
    N[k] = []
    for i in instance:
        if instance[i][0] == k:
            N[k].append(i)

# save parameterfile in correct format
data = open('usps_d.param','w')
data.write('d\n')
data.write(str(number_of_classes)+'\n')
data.write(str(number_of_dimensions)+'\n')
for k in range(1,number_of_classes+1,1):
    data.write(str(k)+'\n')
    data.write(str(prior[k])+'\n')
    for d in range(1,number_of_dimensions+1,1):
        data.write(str(mean[k][d]) + ' ')
    data.write('\n')
    for d in range(1,number_of_dimensions+1,1):
        data.write(str(variance[d]))
    data.write('\n')
data.close()

# d)
def classify(number_of_classes,instance,mean,N,covariance):
    r = {} # assign a vector (instancenumber) to a class
    M = {} # confusion matrix
    # initialize confusion matrix
    for k in range(1,number_of_classes+1,1):
        for j in range(1,number_of_classes+1,1):
            M[(k,j)] = 0
    wrong_classifications = 0
    # use the classifier to assign each vector to a class
    for i in instance:
        argmax_k = 0
        max_k = 0
        for k in range(1,number_of_classes+1,1):
            diffvector = {} # difference x-mu_k
            for d in range(1,number_of_dimensions+1,1):
                diffvector[d] = instance[i][1][d] - mean[k][d]
            exponent = 0 # vector multiplication, scaling with Sigma^-1
            for d in range(1,number_of_dimensions+1,1):
                exponent += 1/covariance[d,d] * (diffvector[d] ** 2)
            value = len(N[k]) * numpy.exp(- 1/2 * exponent)
            if value >= max_k: # find best value/best fitting class
                argmax_k = k
                max_k = value
        r[i] = argmax_k # assign instancenumber to class
        if r[i] != instance[i][0]:
            wrong_classifications += 1
        M[(instance[i][0],r[i])] += 1
    empirical_error_rate = wrong_classifications/len(instance)
    return(empirical_error_rate,M)

# classify usps.train
classify(number_of_classes,instance,mean,N,covariance)

# classify usps.test
inst = read_data('usps.test')
number_of_classes = inst[0]
number_of_dimensions = inst[1]
instance = inst[2]
mean = calculate_mean(number_of_classes,number_of_dimensions,instance)
variance = calculate_variance(number_of_dimensions,instance,mean)
covariance = calculate_covariance(number_of_dimensions,variance)
prior = calculate_p(number_of_classes,instance)
N = {}
for k in range(1,number_of_classes+1,1):
    N[k] = []
    for i in instance:
        if instance[i][0] == k:
            N[k].append(i)
test = classify(number_of_classes,instance,mean,N,covariance)
eer = test[0]
cm = test[1]

# e)
# save error rate of test instance
result_error_rate = open('usps_d.error','w')
result_error_rate.write(str(eer))
result_error_rate.close()

#save confusion matrix of test instance
result_confusion_matrix = open('usps_d.cm','w')
for k in range(1,number_of_classes+1,1):
    for j in range(1,number_of_classes+1,1):
        result_confusion_matrix.write(str(cm[(k,j)]) + '\t')
    result_confusion_matrix.write('\n')
result_confusion_matrix.close()
