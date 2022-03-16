import numpy as np
from BNGBS_DEMO.Algorithm.transfer_increment import GradientBoostingNet
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

print('/*=========================DIG dataset===============================*/')
traindata1 = np.load(path+'/Data/DIG_transfer_increment/DIG_traindata1.npy')
trainlabel1 = np.load(path+'/Data/DIG_transfer_increment/DIG_trainlabel1.npy')
traindata2 = np.load(path+'/Data/DIG_transfer_increment/DIG_traindata2.npy')
trainlabel2 = np.load(path+'/Data/DIG_transfer_increment/DIG_trainlabel2.npy')
testdata1 = np.load(path+'/Data/DIG_transfer_increment/DIG_testdata1.npy')
testlabel1 = np.load(path+'/Data/DIG_transfer_increment/DIG_testlabel1.npy')
testdata2 = np.load(path+'/Data/DIG_transfer_increment/DIG_testdata2.npy')
testlabel2 = np.load(path+'/Data/DIG_transfer_increment/DIG_testlabel2.npy')
print("basic information of DIG dataset:")
print("the first batch training data's shape  : ", traindata1.shape)
print("the second batch training data's shape : ", traindata2.shape)
print("the test data1's shape                  : ", testdata1.shape)
print("the test data2's shape                  : ", testdata2.shape)

print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(learning_rate = 1e-3,
                          n_estimators = 10,
                          maptimes = 2, 
                          enhencetimes = 2,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 500, 
                          reg = 1)

gbn.fit(traindata1,trainlabel1)
predictlabel = gbn.predict(testdata1)
print()
print('the accuracy of BBNGBS on testdata1 is {0}'.format(show_accuracy(predictlabel,testlabel1)))
print()

gbn.adding_class(traindata1,trainlabel1,traindata2,trainlabel2)
predictlabel = gbn.predict(testdata2)

print('the accuracy of BNGBS on testdata2 is {0}'.format(show_accuracy(predictlabel,testlabel2)))

print('/*=========================YAL dataset===============================*/')
traindata1 = np.load(path+'/Data/YAL_transfer_increment/YAL_traindata1.npy')
trainlabel1 = np.load(path+'/Data/YAL_transfer_increment/YAL_trainlabel1.npy')
traindata2 = np.load(path+'/Data/YAL_transfer_increment/YAL_traindata2.npy')
trainlabel2 = np.load(path+'/Data/YAL_transfer_increment/YAL_trainlabel2.npy')
testdata1 = np.load(path+'/Data/YAL_transfer_increment/YAL_testdata1.npy')
testlabel1 = np.load(path+'/Data/YAL_transfer_increment/YAL_testlabel1.npy')
testdata2 = np.load(path+'/Data/YAL_transfer_increment/YAL_testdata2.npy')
testlabel2 = np.load(path+'/Data/YAL_transfer_increment/YAL_testlabel2.npy')
print("basic information of YAL dataset:")
print("the first batch training data's shape  : ", traindata1.shape)
print("the second batch training data's shape : ", traindata2.shape)
print("the test data1's shape                  : ", testdata1.shape)
print("the test data2's shape                  : ", testdata2.shape)

print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(learning_rate = 1e-3,
                          n_estimators = 10,
                          maptimes = 10, 
                          enhencetimes = 10,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 100, 
                          reg = 1)

gbn.fit(traindata1,trainlabel1)
predictlabel = gbn.predict(testdata1)
print()
print('the accuracy of BNGBS on testdata1 is {0}'.format(show_accuracy(predictlabel,testlabel1)))
print()

gbn.adding_class(traindata1,trainlabel1,traindata2,trainlabel2)
predictlabel = gbn.predict(testdata2)
print('the accuracy of BNGBS on testdata2 is {0}'.format(show_accuracy(predictlabel,testlabel2)))

print('/*=========================SPF dataset===============================*/')
traindata1 = np.load(path+'/Data/SPF_transfer_increment/SPF_traindata1.npy')
trainlabel1 = np.load(path+'/Data/SPF_transfer_increment/SPF_trainlabel1.npy')
traindata2 = np.load(path+'/Data/SPF_transfer_increment/SPF_traindata2.npy')
trainlabel2 = np.load(path+'/Data/SPF_transfer_increment/SPF_trainlabel2.npy')
testdata1 = np.load(path+'/Data/SPF_transfer_increment/SPF_testdata1.npy')
testlabel1 = np.load(path+'/Data/SPF_transfer_increment/SPF_testlabel1.npy')
testdata2 = np.load(path+'/Data/SPF_transfer_increment/SPF_testdata2.npy')
testlabel2 = np.load(path+'/Data/SPF_transfer_increment/SPF_testlabel2.npy')
print("basic information of SPF dataset:")
print("the first batch training data's shape  : ", traindata1.shape)
print("the second batch training data's shape : ", traindata2.shape)
print("the test data1's shape                  : ", testdata1.shape)
print("the test data2's shape                  : ", testdata2.shape)

print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(learning_rate = 1e-3,
                          n_estimators = 10,
                          maptimes = 20, 
                          enhencetimes = 20,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 50, 
                          reg = 1)

gbn.fit(traindata1,trainlabel1)
predictlabel = gbn.predict(testdata1)
print()
print('the accuracy of BNGBS on testdata1 is {0}'.format(show_accuracy(predictlabel,testlabel1)))
print()
gbn.adding_class(traindata1,trainlabel1,traindata2,trainlabel2)
predictlabel = gbn.predict(testdata2)
print('the accuracy of BNGBS on testdata2 is {0}'.format(show_accuracy(predictlabel,testlabel2)))

