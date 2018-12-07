import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


class Dataset:
    def __init__(self):
        self.index = 0

        self.obs = []
        self.classes = []
        self.num_obs = 0
        self.num_classes = 0
        self.indices = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_obs:
            self.index = 0
            raise StopIteration
        else:
            self.index += 1
            return self.obs[self.index - 1], self.classes[self.index - 1]

    def reset(self):
        self.index = 0

    def get_obs_with_target(self, k):
        index_list = [index for index, value in enumerate(self.classes) if value == k]
        return [self.obs[i] for i in index_list]

    def get_all_obs_class(self, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)
        return [(self.obs[i], self.classes[i]) for i in self.indices]

    def get_mini_batches(self, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)

        batches = [(self.obs[self.indices[n:n + batch_size]],
                    self.classes[self.indices[n:n + batch_size]])
                   for n in range(0, self.num_obs, batch_size)]
        return batches


class IrisDataset(Dataset):
    def __init__(self, path):
        super(IrisDataset, self).__init__()
        self.file_path = path
        self.loadFile()
        self.indices = np.arange(self.num_obs)

    def loadFile(self):
        # load a comma-delimited text file into an np matrix
        resultList = []
        f = open(self.file_path, 'r')
        for line in f:
            line = line.rstrip('\n')  # "1.0,2.0,3.0"
            sVals = line.split(',')  # ["1.0", "2.0, "3.0"]
            fVals = list(map(np.float32, sVals))  # [1.0, 2.0, 3.0]
            resultList.append(fVals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
        f.close()
        data = np.asarray(resultList, dtype=np.float32)  # not necessary
        self.obs = data[:, 0:4]
        self.classes = data[:, 4:7]
        self.num_obs = data.shape[0]
        self.num_classes = 3


# Activations
def tanh(x, deriv=False):
    '''
	d/dx tanh(x) = 1 - tanh^2(x)
	during backpropagation when we need to go though the derivative we have already computed tanh(x),
	therefore we pass tanh(x) to the function which reduces the gradient to:
	1 - tanh(x)
    '''
    if deriv:
        return 1.0 - np.tanh(x) # Korrektur in Glob: 1.0 - x^2
    else:
        return np.tanh(x)


def sigmoid(x, deriv=False):
    '''
    Task 2a
    This function is the sigmoid function. It gets an input digit or vector and should return sigmoid(x).
    The parameter "deriv" toggles between the sigmoid and the derivate of the sigmoid. Hint: In the case of the derivate
    you can expect the input to be sigmoid(x) instead of x
    :param x:
    :param deriv:
    :return:
    '''
    if deriv :
        return (np.reciprocal(x) - 1)*np.power(x,2)
    else :
        return np.reciprocal(1 + np.exp(-x))


def softmax(x, deriv=False):
    '''
    Task 2a
    This function is the sigmoid function with a softmax applied. This will be used in the last layer of the network
    The derivate will be the same as of sigmoid(x)
    :param x:
    :param deriv:
    :return:
    '''
    if deriv : 
        # x + np.power(x,2) is the same as
        return sigmoid(x,deriv)
    else :
        return np.exp(x)/np.sum(np.exp(x)) 


class Layer:
    def __init__(self, numInput, numOutput, activation=sigmoid):
        print('Create layer with: {}x{} @ {}'.format(numInput, numOutput, activation))
        self.ni = numInput
        self.no = numOutput
        self.weights = np.zeros(shape=[self.ni, self.no], dtype=np.float32)
        self.biases = np.zeros(shape=[self.no], dtype=np.float32)
        self.initializeWeights()

        self.activation = activation
        self.last_input = None	# placeholder, can be used in backpropagation
        self.last_output = None # placeholder, can be used in backpropagation
        self.last_nodes = None  # placeholder, can be used in backpropagation

    def initializeWeights(self):
        """
        Task 2d
        Initialized the weight matrix of the layer. Weights should be initialized to something other than 0.
        You can search the literature for possible initialization methods.
        :return: None
        """
        for i in self.ni :
            for j in self.no :
                while self.weights[i,j] == 0 or self.biases[j] == 0 :
                    self.weights[i,j] = np.randf()
                    self.biases[j] = np.randf()

    def inference(self, x):
        """
        Task 2b
        This transforms the input x with the layers weights and bias and applies the activation function
        Hint: you should save the input and output of this function usage in the backpropagation
        :param x:
        :return: output of the layer
        :rtype: np.array
        """
        self.last_input = x #y^(l-1)
        y = np.zeros(self.no)
        for i in range(self.no) :
            sum = self.biases[i]
            for j in range(self.ni) :
                sum += self.weights[j,i]*x[j] 
            y[i] = self.activation(sum)
        self.last_input = y  #sig(z_i)  
        return y
        

    def backprop(self, error):
        """
        Task 2c
        This function applied the backpropagation of the error signal. The Layer receives the error signal from the following
        layer or the network. You need to calculate the error signal for the next layer by backpropagating thru this layer.
         You also need to compute the gradients for the weights and bias.
        :param error:
        :return: error signal for the preceeding layer, called ep
        :return: gradients for the weight matrix, called ew
        :return: gradients for the bias, called eb
        :rtype: np.array
        """
        ew = np.zeros(shape=[self.ni, self.no], dtype=np.float32)
        for i in range(self.ni) :
            for j in range(self.no) :
                ew[i,j] = error(i)*sigmoid(self.last_input[i], True)*self.last_output[j]
        
        ep = np.zeros(self.ni)
        for i in range(self.ni) :
            sum = 0
            for k in range(self.no) :
                sum += error[k]*sigmoid(self.last_input[k], True)*self.weights[k,i]
            ep[i] = sum
            
        eb = error*sigmoid(self.last_input, True)
     

        return ep, ew, eb


class BasicNeuralNetwork():
    def __init__(self, layer_sizes=[5], num_input=4, num_output=3, num_epoch=50, learning_rate=0.1,
                 mini_batch_size=8):
        self.layers = []
        self.ls = layer_sizes
        self.ni = num_input
        self.no = num_output
        self.lr = learning_rate
        self.num_epoch = num_epoch
        self.mbs = mini_batch_size

        self.constructNetwork()

    def forward(self, x):
        """
        Task 2b
        This function forwards a single feature vector through every layer and return the output of the last layer
        :param x: input feature vector
        :return: output of the network
        :rtype: np.array
        """
        x = sigmoid(x)
        for l in self.layers :
            x = l.inference(x)
        return x

    def train(self, train_dataset, eval_dataset=None, monitor_ce_train=True, monitor_accuracy_train=True,
              monitor_ce_eval=True, monitor_accuracy_eval=True, monitor_plot='monitor.png'):
        ce_train_array = []
        ce_eval_array = []
        acc_train_array = []
        acc_eval_array = []
        for e in range(self.num_epoch):
            if self.mbs:
                self.mini_batch_SGD(train_dataset)
            else:
                self.online_SGD(train_dataset)
            print('Finished training epoch: {}'.format(e))
            if monitor_ce_train:
                ce_train = self.ce(train_dataset)
                ce_train_array.append(ce_train)
                print('CE (train): {}'.format(ce_train))
            if monitor_accuracy_train:
                acc_train = self.accuracy(train_dataset)
                acc_train_array.append(acc_train)
                print('Accuracy (train): {}'.format(acc_train))
            if monitor_ce_eval:
                ce_eval = self.ce(eval_dataset)
                ce_eval_array.append(ce_eval)
                print('CE (eval): {}'.format(ce_eval))
            if monitor_accuracy_eval:
                acc_eval = self.accuracy(eval_dataset)
                acc_eval_array.append(acc_eval)
                print('Accuracy (eval): {}'.format(acc_eval))

        if monitor_plot:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            line1, = ax[0].plot(ce_train_array, '--', linewidth=2, label='ce_train')
            line2, = ax[0].plot(ce_eval_array, label='ce_eval')

            line3, = ax[1].plot(acc_train_array, '--', linewidth=2, label='acc_train')
            line4, = ax[1].plot(acc_eval_array, label='acc_eval')

            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper left')
            ax[1].set_ylim([0, 1])

            plt.savefig(monitor_plot)

    def online_SGD(self, dataset):
        """
        Task 2d
        This function trains the network in an online fashion. Meaning the weights are updated after each observation.
        :param dataset:
        :return: None
        """
        gamma = 1
        
        for x,t in dataset :
            ll = self.layers[-1]
            v = forward(self,x)
            v[t] = v[t] - 1                     # delta(k,cn)
            ep, ew, eb = ll.backprob(self,v)
            ll.weights = ll.weights - gamma * ew
            ll.biases = ll.biases - gamma * eb
            for l in range(self.layers-1,-1,-1) :
                temp = ep
                ep, ew, eb = l.backprob(self, temp)
                l.weights = l.weights - gamma * ew
                l.biases = l.biases - gamma * eb
            
        

    def mini_batch_SGD(self, dataset):
        """
        Task 2d
        This function trains the network using mini batches. Meaning the weights updates are accumulated and applied after each mini batch.
        :param dataset:
        :return: None
        """
        gamma = 1
        batch_size = 3
        
        batches = dataset.get_mini_batches(self, batch_size)
        for b in batches :
            mew = []
            meb = []
            for l in self.layers:
                mew_l = mew.append(np.zeros(shape=[l.ni, l.no]))
                meb_l = meb.append(np.zeros(l.no))
                
                
            for x,t in b :
                
                ll = self.layers[-1]
                v = forward(self,x)
                v[t] = v[t] - 1                     # delta(k,cn)
                ep, ew, eb = ll.backprob(self,v)
                mew[-1] = mew[-1] + ew
                meb[-1] = meb[-1] + eb
                
                for l in range(self.layers-1,-1,-1) :
                    temp = ep
                    ep, ew, eb = l.backprob(self, temp)
                    mew[l] = mew[l] + ew
                    meb[l] = meb[l] + eb
        
        for l in self.layers :
            l.weights = l.weights - gamma * mew[l]
            l.biases = l.biases - gamma * meb[l]
            
            

    def constructNetwork(self):
        """
        Task 2d
        uses self.ls self.ni and self.no to construct a list of layers. The last layer should use sigmoid_softmax as an activation function. any preceeding layers should use sigmoid.
        :return: None
        """
        pass

    def ce(self, dataset):
        ce = 0
        for x, t in dataset:
            t_hat = self.forward(x)
            ce += np.sum(np.nan_to_num(-t * np.log(t_hat) - (1 - t) * np.log(1 - t_hat)))

        return ce / dataset.num_obs

    def accuracy(self, dataset):
        cm = np.zeros(shape=[dataset.num_classes, dataset.num_classes], dtype=np.int)
        for x, t in dataset:
            t_hat = self.forward(x)
            c_hat = np.argmax(t_hat)  # index of largest output value
            c = np.argmax(t)
            cm[c, c_hat] += 1

        correct = np.trace(cm)
        return correct / dataset.num_obs

    def load(self, path=None):
        if not path:
            path = './network.save'
        with open(path, 'rb') as f:
            self.layers = pickle.load(f)

    def save(self, path=None):
        if not path:
            path = './network.save'
        with open(path, 'wb') as f:
            pickle.dump(self.layers, f)
