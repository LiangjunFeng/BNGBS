import numpy as np
from sklearn import preprocessing
from numpy import random
import pandas as pd
import random as Random 
from scipy.linalg import orth
import datetime
import gc  

def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count/len(Label),5))

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0
    
    def fit_transform(self,traindata):
        self._mean = traindata.mean(axis = 0)
        self._std = traindata.std(axis = 0)
        self._std += 0.1
        return (traindata-self._mean)/self._std
        
    def transform(self,testdata):
        return (testdata-self._mean)/self._std


class node_generator:
    def __init__(self,whiten = True):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten
    
    def sigmoid(self,data):
        return 1.0/(1+np.exp(-data))
    
    def linear(self,data):
        data -= np.min(data) 
        return data/np.max(data)
    
    def tanh(self,data):
        return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    
    def relu(self,data):
        return np.maximum(data,0)
    
    def bgenerator(self,shape,times):
        for i in range(times):
            b = 2*random.random()-1
            yield b

    def wgenerator(self,shape,times):
        for i in range(times):
            W = orth(2*random.random(size=shape)-1)
            yield W

    def generator_nodes(self, data, times, batchsize, nonlinear):
        self.Wlist = [elem for elem in self.wgenerator((data.shape[1],batchsize),times)]
        self.blist = [elem for elem in self.bgenerator((data.shape[1],batchsize),times)]
        self.nonlinear = {'linear':self.linear,
                          'sigmoid':self.sigmoid,
                          'tanh':self.tanh,
                          'relu':self.relu
                          }[nonlinear]
        
        nodes = self.nonlinear(data.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i])+self.blist[i])))
        return nodes
        
    def transform(self,testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i])+self.blist[i])))
        return testnodes  


class broadnet_enhmap:
    def __init__(self, 
                 maptimes = 10, 
                 enhencetimes = 10,
                 map_function = 'linear',
                 enhence_function = 'linear',
                 batchsize = 'auto', 
                 reg = 0.001):
        
        self._maptimes = maptimes
        self._enhencetimes = enhencetimes
        self._batchsize = batchsize
        self._reg = reg
        self._map_function = map_function
        self._enhence_function = enhence_function 
        self.W = 0
        self.pesuedoinverse = 0
        
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator()

    def fit(self,data,label):

        
        mappingdata = self.mapping_generator.generator_nodes(data,self._maptimes,self._batchsize,self._map_function)
        enhencedata = self.enhence_generator.generator_nodes(mappingdata,self._enhencetimes,self._batchsize,self._enhence_function)
        inputdata = np.column_stack((mappingdata,enhencedata))

        self.pesuedoinverse = self.pinv(inputdata)
        self.W =  self.pesuedoinverse.dot(label)
        
    def pinv(self, A):
        return np.mat(self._reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
  
    def predict_value(self,testdata):
        test_inputdata = self.transform(testdata) 
        return test_inputdata.dot(self.W)    
        
    def transform(self,data):
        mappingdata = self.mapping_generator.transform(data)
        enhencedata = self.enhence_generator.transform(mappingdata)
        inputdata = np.column_stack((mappingdata,enhencedata))
        return inputdata 
    
class GradientBoostingNet:
    def __init__(self,
                 column_sampling = 1, 
                 row_sampling = 1, 
                 learning_rate = 0.001,
                 n_estimators = 10,
                 maptimes = 10, 
                 enhencetimes = 10,
                 map_function = 'linear',
                 enhence_function = 'linear',
                 batchsize = 'auto', 
                 reg = 0.001):
        
        self._column_sampling = column_sampling
        self._row_sampling = row_sampling
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        
        self._maptimes = maptimes
        self._enhencetimes = enhencetimes
        self._map_function = map_function
        self._enhence_function = enhence_function
        self._batchsize = batchsize
        self._reg = reg
        
        self.onehot = preprocessing.OneHotEncoder(sparse = False)
        self.scaler = scaler()
        self.f = 0
        self.net_list = []
        self.column_llist = []
            
    def softmax(self,train_value):
        temp1 = np.exp(train_value)
        temp2 = np.mat(np.sum(temp1,axis=1))
        if temp1.shape[0] != temp2.shape[0]:
            temp2 = temp2.T
        return temp1 / temp2
     
    def sampling(self,traindata,trainlabel,row_sample,column_sample):
        column_list = list(range(traindata.shape[1]))
        row_list = list(range(traindata.shape[0]))
         
        column_samplelist = Random.sample(column_list,int(len(column_list)*column_sample))
        data = traindata[column_samplelist]
        self.column_llist.append(column_samplelist)
        
        row_sampleist = Random.sample(row_list,int(len(row_list)*row_sample))
        row_remainlist = list(set(row_list)^set(row_sampleist))
        
        re_data = data.iloc[row_sampleist]
        re_label = trainlabel.iloc[row_sampleist]
        ex_data = data.iloc[row_remainlist]
        ex_label = trainlabel.iloc[row_remainlist]
        
        del row_sampleist[:]
        del row_remainlist[:]
        gc.collect()
        
        return re_data,re_label,ex_data,ex_label
        
    def stack(self,result1,result2,data1,data2):
        result1 = pd.DataFrame(result1)
        result2 = pd.DataFrame(result2)
        result1['index'] = data1.index
        result1 = result1.set_index('index')
        result2['index'] = data2.index
        result2 = result2.set_index('index')
        
        res = pd.DataFrame(pd.concat((result1,result2)))
        return res.sort_index().values
    
    def prepocess(self,traindata,trainlabel):
        label_onehot = self.onehot.fit_transform(np.mat(trainlabel).T)    
        traindata = self.scaler.fit_transform(traindata)        
        return pd.DataFrame(traindata),pd.DataFrame(label_onehot)

    def fit(self,traindata,trainlabel):
        starttime = datetime.datetime.now()
        if self._batchsize == 'auto':
            self._batchsize = traindata.shape[1]
        data,label = self.prepocess(traindata,trainlabel)
        self.f = np.zeros(label.shape)
        orilabel = label.copy()
        for i in range(self._n_estimators):
            data1,label1,data2,label2 = self.sampling(data,label,self._row_sampling,self._column_sampling)
            base_net = broadnet_enhmap(self._maptimes,
                                       self._enhencetimes,
                                       self._map_function,
                                       self._enhence_function,
                                       self._batchsize,
                                       self._reg)

            base_net.fit(data1.values,label1.values)
            
            result1 = base_net.predict_value(data1.values)
            result2 = base_net.predict_value(data2.values)
            result = self.stack(result1,result2,data1,data2)
            print('the {0}th base learner is training, the number of mapping nodes is {1}, the number of mapping nodes is {2}'.format(i+1,self._maptimes*self._batchsize,self._enhencetimes*self._batchsize))
            
            self.f += self._learning_rate * result
            self.net_list.append(base_net)
            label = orilabel - self.softmax(self.f)

        endtime = datetime.datetime.now()
        print()
        print('the training time of BNGBS is {0} seconds'.format((endtime - starttime).total_seconds()))
 
    def predict_proba(self,testdata):
        testdata = self.scaler.transform(testdata)
        f = np.zeros((testdata.shape[0],self.f.shape[1]))
        
        for i in range(self._n_estimators):
            data = testdata[:,self.column_llist[i]]
            net = self.net_list[i]
            f += net.predict_value(data)
            
        return self.softmax(f)
    
    def predict(self,testdata):
        t = self.predict_proba(testdata)
        return self.decode(t)
     
    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i,:]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)       

def softmax(train_value):
    temp1 = np.exp(train_value)
    temp2 = np.mat(np.sum(temp1,axis=1))
    if temp1.shape[0] != temp2.shape[0]:
        temp2 = temp2.T
    return temp1 / temp2