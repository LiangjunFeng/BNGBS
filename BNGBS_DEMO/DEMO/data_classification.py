import numpy as np
from BNGBS_DEMO.Algorithm.data_increment import GradientBoostingNet
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

print('/*======================DIG dataset============================*/')
data1 = np.load(path+'/Data/DIG_data_increment/DIG_data1.npy')
label1 = np.load(path+'/Data/DIG_data_increment/DIG_label1.npy')
data2 = np.load(path+'/Data/DIG_data_increment/DIG_data2.npy')
label2 = np.load(path+'/Data/DIG_data_increment/DIG_label2.npy')
testdata = np.load(path+'/Data/DIG_data_increment/DIG_testdata.npy')
testlabel = np.load(path+'/Data/DIG_data_increment/DIG_testlabel.npy')

print("basic information of DIG dataset:")
print("the first batch training data's shape  : ", data1.shape)
print("the second batch training data's shape : ", data2.shape)
print("the test data's shape                  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(learning_rate = 1e-3,
                          n_estimators = 10,
                          maptimes = 1, 
                          enhencetimes = 1,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 1000, 
                          reg = 1)

gbn.fit(data1,label1)
predictlabel = gbn.predict(testdata)
print()
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()

gbn.adding_data(data1,label1,data2,label2)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))

print('/*======================YAL dataset============================*/')
data1 = np.load(path+'/Data/YAL_data_increment/YAL_data1.npy')
label1 = np.load(path+'/Data/YAL_data_increment/YAL_label1.npy')
data2 = np.load(path+'/Data/YAL_data_increment/YAL_data2.npy')
label2 = np.load(path+'/Data/YAL_data_increment/YAL_label2.npy')
testdata = np.load(path+'/Data/YAL_data_increment/YAL_testdata.npy')
testlabel = np.load(path+'/Data/YAL_data_increment/YAL_testlabel.npy')

print("basic information of YAL dataset: (Processed by PCA)")
print("the first batch training data's shape  : ", data1.shape)
print("the second batch training data's shape : ", data2.shape)
print("the test data's shape                  : ", testdata.shape)
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

gbn.fit(data1,label1)
predictlabel = gbn.predict(testdata)
print()
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()

gbn.adding_data(data1,label1,data2,label2)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))

print('/*======================SPF dataset============================*/')
data1 = np.load(path+'/Data/SPF_data_increment/SPF_data1.npy')
label1 = np.load(path+'/Data/SPF_data_increment/SPF_label1.npy')
data2 = np.load(path+'/Data/SPF_data_increment/SPF_data2.npy')
label2 = np.load(path+'/Data/SPF_data_increment/SPF_label2.npy')
testdata = np.load(path+'/Data/SPF_data_increment/SPF_testdata.npy')
testlabel = np.load(path+'/Data/SPF_data_increment/SPF_testlabel.npy')

print("basic information of SPF dataset: ")
print("the first batch training data's shape  : ", data1.shape)
print("the second batch training data's shape : ", data2.shape)
print("the test data's shape                  : ", testdata.shape)
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

gbn.fit(data1,label1)
predictlabel = gbn.predict(testdata)
print()
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()

gbn.adding_data(data1,label1,data2,label2)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))





















