import numpy as np
from BNGBS_DEMO.Algorithm.node_increment import GradientBoostingNet
import os

path = os.path.dirname(os.getcwd())

def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count/len(Label),5))

print('/*===================DIG dataset=========================*/')
traindata = np.load(path+'/Data/DIG/DIG_traindata.npy')
trainlabel = np.load(path+'/Data/DIG/DIG_trainlabel.npy')
testdata = np.load(path+'/Data/DIG/DIG_testdata.npy')
testlabel = np.load(path+'/Data/DIG/DIG_testlabel.npy')
print("basic information of DIG dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(learning_rate = 1e-3,
                          n_estimators = 10,
                          maptimes = 1, 
                          enhencetimes = 1,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 10, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of GBNet on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()
batchsize = 10
for i in range(8):
    gbn.adding_node(traindata,trainlabel,mapstep = 1,enhencestep = 1,batchsize = batchsize)
    predictlabel = gbn.predict(testdata)
    print('the accuracy of GBNet on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
    batchsize += 10

print('/*===================YAL dataset=========================*/')
traindata = np.load(path+'/Data/YAL/YAL_traindata.npy')
trainlabel = np.load(path+'/Data/YAL/YAL_trainlabel.npy')
testdata = np.load(path+'/Data/YAL/YAL_testdata.npy')
testlabel = np.load(path+'/Data/YAL/YAL_testlabel.npy')
print("basic information of YAL dataset (Processed by PCA):")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()
gbn = GradientBoostingNet(learning_rate = 1e-3,
                          n_estimators = 10,
                          maptimes = 1, 
                          enhencetimes = 1,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 10, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of GBNet on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()
batchsize = 10
for i in range(8):
    gbn.adding_node(traindata,trainlabel,mapstep = 1,enhencestep = 1,batchsize = batchsize)
    predictlabel = gbn.predict(testdata)
    print('the accuracy of GBNet on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
    batchsize += 10
 
    
print('/*===================SPF dataset=========================*/')
traindata = np.load(path+'/Data/SPF/SPF_traindata.npy')
trainlabel = np.load(path+'/Data/SPF/SPF_trainlabel.npy')
testdata = np.load(path+'/Data/SPF/SPF_testdata.npy')
testlabel = np.load(path+'/Data/SPF/SPF_testlabel.npy')
print("basic information of SPF dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()  
gbn = GradientBoostingNet(learning_rate = 1e-3,
                          n_estimators = 10,
                          maptimes = 1, 
                          enhencetimes = 1,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 10, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of GBNet on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()
batchsize = 10
for i in range(8):
    gbn.adding_node(traindata,trainlabel,mapstep = 1,enhencestep = 1,batchsize = batchsize)
    predictlabel = gbn.predict(testdata)
    print('the accuracy of GBNet on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
    batchsize += 10
    
    
    
    
    
    
    
    
    
    
    
    
    
    