import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy.linalg import orth
import datetime

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
            W = orth((2*random.random(size=shape)-1))
            if W.shape != shape:
                W = orth((2*random.random(size=shape)-1).T).T
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
        self.local_mappinglist = []
        self.local_enhencelist = []    
        
        self.mappingnodes_number = 0
        self.enhencenodes_number = 0

    def fit(self,data,label):
        if self._batchsize == 'auto':
            self._batchsize = data.shape[1]
        
        mappingdata = self.mapping_generator.generator_nodes(data,self._maptimes,self._batchsize,self._map_function)
        enhencedata = self.enhence_generator.generator_nodes(mappingdata,self._enhencetimes,self._batchsize,self._enhence_function)
        inputdata = np.column_stack((mappingdata,enhencedata))
        
        self.mappingnodes_number += mappingdata.shape[1]
        self.enhencenodes_number += enhencedata.shape[1]

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
        for elem1,elem2 in zip(self.local_mappinglist,self.local_enhencelist):
            inputdata  = np.column_stack((inputdata, elem1.transform(data)))
            inputdata  = np.column_stack((inputdata, elem2.transform(mappingdata)))
        return inputdata
    
    def mapping_transform(self,data):
        mappingdata = self.mapping_generator.transform(data)
        return mappingdata
            
    def adding_node(self, data, label, mapstep = 1, enhencestep = 1, batchsize = 1):  
        mappingdata = self.mapping_transform(data)
        inputdata = self.transform(data)
        
        local_mappinggenerator = node_generator()
        local_enhencegenerator = node_generator()
        

        extra_mappingdata = local_mappinggenerator.generator_nodes(data,mapstep,batchsize,self._map_function)
        extra_enhencedata = local_enhencegenerator.generator_nodes(mappingdata,enhencestep,batchsize,self._enhence_function)   

      
        extra_data = np.column_stack((extra_mappingdata,extra_enhencedata))
        
        self.mappingnodes_number += extra_mappingdata.shape[1]
        self.enhencenodes_number += extra_enhencedata.shape[1]
        
        D = self.pesuedoinverse.dot(extra_data)
        C = extra_data - inputdata.dot(D)
        BT = self.pinv(C) if (C != 0).any() else np.mat((D.T.dot(D)+np.eye(D.shape[1]))).I.dot(D.T).dot(self.pesuedoinverse)
        
        self.pesuedoinverse =  np.row_stack((self.pesuedoinverse - D.dot(BT),BT)) 
        self.W = self.pesuedoinverse.dot(label)
         
        self.local_mappinglist.append(local_mappinggenerator)
        self.local_enhencelist.append(local_enhencegenerator)   
    
    def label_update(self,new_label,old_label):
        self.W = self.pesuedoinverse.dot(new_label)

class GradientBoostingNet:
    def __init__(self,
                 learning_rate = 0.001,
                 n_estimators = 10,
                 maptimes = 10, 
                 enhencetimes = 10,
                 map_function = 'linear',
                 enhence_function = 'linear',
                 batchsize = 'auto', 
                 reg = 0.001):
    
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
        self.label_list = []
            
    def softmax(self,train_value):
        temp1 = np.exp(train_value)
        temp2 = np.mat(np.sum(temp1,axis=1))
        if temp1.shape[0] != temp2.shape[0]:
            temp2 = temp2.T
        return temp1 / temp2
    
    def prepocess(self,traindata,trainlabel):
        label_onehot = self.onehot.fit_transform(np.mat(trainlabel).T)    
        traindata = self.scaler.fit_transform(traindata)        
        return traindata,label_onehot

    def fit(self,traindata,trainlabel):
        starttime = datetime.datetime.now()
        data,label = self.prepocess(traindata,trainlabel)
        self.f = np.zeros(label.shape)
        orilabel = label.copy()
        
        for i in range(self._n_estimators):
            self.label_list.append(np.mat(label))           
            base_net = broadnet_enhmap(self._maptimes,
                                       self._enhencetimes,
                                       self._map_function,
                                       self._enhence_function,
                                       self._batchsize,
                                       self._reg)
            base_net.fit(data,label)
            result = base_net.predict_value(data)

            self.f += self._learning_rate * result
            self.net_list.append(base_net)
            label = orilabel - self.softmax(self.f)            
            print('the {0}th base learner is training, the number of mapping nodes is {1}, the number of enhence nodes is {2}'.format(i+1,self.net_list[0].mappingnodes_number,self.net_list[0].enhencenodes_number))
            

            
        endtime = datetime.datetime.now()
        print('the training time of BNGBS is {0} seconds'.format((endtime - starttime).total_seconds()))
        
    def adding_node(self, data, label, mapstep = 1, enhencestep = 1 , batchsize= 1):
        
        starttime = datetime.datetime.now()
        label = self.onehot.transform(np.mat(label).T)
        data = self.scaler.transform(data)
        f = np.zeros(label.shape)
        
        new_label = label.copy()
        for i in range(self._n_estimators):
            net = self.net_list[i]
            net.adding_node(data,label,mapstep,enhencestep,batchsize)
            
            trainlabel = self.label_list[i]
            net.label_update(new_label,trainlabel)
            result = net.predict_value(data)
            f += self._learning_rate * result
            new_label = label - self.softmax(f)
        endtime = datetime.datetime.now()
        print('the training time of adding is {0} seconds, the number of mapping nodes is {1}, the number of enhence nodes is {2}'.format((endtime - starttime).total_seconds(),self.net_list[0].mappingnodes_number,self.net_list[0].enhencenodes_number))
    def predict_proba(self,testdata):
        testdata = self.scaler.transform(testdata)
        f = np.zeros((testdata.shape[0],self.f.shape[1]))
        
        for i in range(self._n_estimators):
            net = self.net_list[i]
            f += net.predict_value(testdata)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    