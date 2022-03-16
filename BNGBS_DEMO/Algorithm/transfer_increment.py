import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy.linalg import orth
import datetime
import copy 

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
        
    def transform_all(self,data,xdata_list):
        inputdata = self.transform(data)
        for elem in xdata_list:
            inputdata = np.row_stack((inputdata,self.transform(elem)))
        return inputdata
    
    def transform(self,data):
        mappingdata = self.mapping_generator.transform(data)
        enhencedata = self.enhence_generator.transform(mappingdata)
        inputdata = np.column_stack((mappingdata,enhencedata))
        return inputdata
    
    def mapping_transform(self,data):
        mappingdata = self.mapping_generator.transform(data)
        return mappingdata

    def adding_class(self, data, xdata_list, xdata, xlabel):  

        data = self.transform_all(data,xdata_list)
        xdata = self.transform(xdata).T
        xlabel = xlabel.T
        
        DT = xdata.T.dot(self.pesuedoinverse)
        CT = xdata.T - DT.dot(data)
        B = self.pinv(CT) if (CT != 0).any() else (np.mat((DT.dot(DT.T)+np.eye(DT.shape[0]))).I).dot(self.pesuedoinverse.dot(DT.T))
        self.W = np.column_stack((self.W,np.zeros((self.W.shape[0],1))))
        self.W = self.W + B*((xlabel.T-(xdata.T)*(self.W)))
        self.pesuedoinverse = np.column_stack((self.pesuedoinverse-B.dot(DT),B))

    def label_update(self,new_label,old_label):
        self.W += self.pesuedoinverse.dot(new_label-old_label)
        

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
        self.xdata_list= []
        self.xlabel_list = []
        self.trainlabel_list = []
        self.newlabel_list =[]
            
    def softmax(self,train_value):
        temp1 = np.exp(train_value)
        temp2 = np.mat(np.sum(temp1,axis=1))
        if temp1.shape[0] != temp2.shape[0]:
            temp2 = temp2.T
        return temp1 / temp2
    
    def prepocess(self,traindata): 
        traindata = self.scaler.fit_transform(traindata)        
        return traindata

    def fit(self,traindata,trainlabel):
        starttime = datetime.datetime.now()
        traindata = self.prepocess(traindata)
        self.f = np.zeros(trainlabel.shape)
        orilabel = trainlabel.copy()
        
        for i in range(self._n_estimators):            
            base_net = broadnet_enhmap(self._maptimes,
                                       self._enhencetimes,
                                       self._map_function,
                                       self._enhence_function,
                                       self._batchsize,
                                       self._reg)
            
            self.trainlabel_list.append(trainlabel)
            base_net.fit(traindata,trainlabel)
            result = base_net.predict_value(traindata)

            self.f += self._learning_rate * result
            self.net_list.append(base_net)
            trainlabel = orilabel - self.softmax(self.f)   
            print('the {0}th base learner is training, the number of mapping nodes is {1}, the number of enhence nodes is {2}'.format(i+1,self.net_list[0].mappingnodes_number,self.net_list[0].enhencenodes_number))
            
        endtime = datetime.datetime.now()
        print('the training time of BNGBS is {0} seconds'.format((endtime - starttime).total_seconds()))
    
    def get_data(self,data,xdata_list,xdata):
        for item in xdata_list:
            data = np.row_stack((data,item))
        return np.row_stack((data,xdata))

    def adding_class(self, data, label, xdata, xlabel):
        
        starttime = datetime.datetime.now()
        data = self.scaler.transform(data)
        xdata = self.scaler.transform(xdata)
        for i in range(self._n_estimators):
            net = self.net_list[i]
            net.adding_class(data, self.xdata_list, xdata,xlabel)

        traindata = self.get_data(data,self.xdata_list,xdata)
        for i in range(self._n_estimators):
            self.trainlabel_list[i] = np.column_stack((self.trainlabel_list[i],np.zeros((self.trainlabel_list[i].shape[0],1))))
        
        new_label = np.row_stack((self.trainlabel_list[0],xlabel))
        self.newlabel_list.append(new_label)
        f = np.zeros(new_label.shape)
        
        for i in range(self._n_estimators):
            trainlabel = np.row_stack((self.trainlabel_list[i],new_label[self.trainlabel_list[i].shape[0]:,:]))
            net = self.net_list[i]
            net.label_update(new_label,trainlabel)
            result = net.predict_value(traindata)
            f += self._learning_rate * result
            new_label = new_label - self.softmax(f)
            self.newlabel_list.append(new_label)

        self.trainlabel_list = copy.deepcopy(self.newlabel_list)
        self.newlabel_list = []
        self.xdata_list.append(xdata)
        self.xlabel_list.append(xlabel)
        endtime = datetime.datetime.now()

        print('the training time of adding is {0} seconds, the number of mapping nodes is {1}, the number of enhence nodes is {2}'.format((endtime - starttime).total_seconds(),self.net_list[0].mappingnodes_number,self.net_list[0].enhencenodes_number))
        
    def predict_proba(self,testdata):
        testdata = self.scaler.transform(testdata)
        f = np.zeros((testdata.shape[0],self.trainlabel_list[0].shape[1]))
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



