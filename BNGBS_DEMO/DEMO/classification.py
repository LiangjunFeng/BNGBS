import numpy as np
from BNGBS_DEMO.Algorithm.BNGBS import GradientBoostingNet
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


print('/*===================BRE dataset=========================*/')
traindata = np.load(path+'/Data/BRE/BRE_traindata.npy')
trainlabel = np.load(path+'/Data/BRE/BRE_trainlabel.npy')
testdata = np.load(path+'/Data/BRE/BRE_testdata.npy')
testlabel = np.load(path+'/Data/BRE/BRE_testlabel.npy')
print("basic information of BRE dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()
gbn = GradientBoostingNet(column_sampling = 1, 
                                 row_sampling = 1, 
                                 learning_rate = 0.1,
                                 n_estimators = 10,
                                 maptimes = 20, 
                                 enhencetimes = 20,
                                 map_function = 'tanh',
                                 enhence_function = 'tanh',
                                 batchsize = 50, 
                                 reg = 1) 
gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))

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

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 20, 
                          enhencetimes = 20,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 50, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))


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

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 10, 
                          enhencetimes = 10,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 100, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))

print('/*===================MPF dataset=========================*/')
traindata = np.load(path+'/Data/MPF/MPF_traindata.npy')
trainlabel = np.load(path+'/Data/MPF/MPF_trainlabel.npy')
testdata = np.load(path+'/Data/MPF/MPF_testdata.npy')
testlabel = np.load(path+'/Data/MPF/MPF_testlabel.npy')
print("basic information of MPF dataset :")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 20, 
                          enhencetimes = 20,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 50, 
                          reg = 1e-3)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))

print('/*===================DIA dataset=========================*/')
traindata = np.load(path+'/Data/DIA/DIA_traindata.npy')
trainlabel = np.load(path+'/Data/DIA/DIA_trainlabel.npy')
testdata = np.load(path+'/Data/DIA/DIA_testdata.npy')
testlabel = np.load(path+'/Data/DIA/DIA_testlabel.npy')
print("basic information of DIA dataset (after removing seven text dims):")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 10, 
                          enhencetimes = 10,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 100, 
                          reg = 1e-3)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))


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

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 50, 
                          enhencetimes = 50,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 20, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))

print('/*===================USP dataset=========================*/')
traindata = np.load(path+'/Data/USP/USP_traindata.npy')
trainlabel = np.load(path+'/Data/USP/USP_trainlabel.npy')
testdata = np.load(path+'/Data/USP/USP_testdata.npy')
testlabel = np.load(path+'/Data/USP/USP_testlabel.npy')
print("basic information of USP dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 20, 
                          enhencetimes = 20,
                          map_function = 'tanh',
                          enhence_function = 'sigmoid',
                          batchsize = 50, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))

print('/*===================CAE dataset=========================*/')
traindata = np.load(path+'/Data/CAE/CAE_traindata.npy')
trainlabel = np.load(path+'/Data/CAE/CAE_trainlabel.npy')
testdata = np.load(path+'/Data/CAE/CAE_testdata.npy')
testlabel = np.load(path+'/Data/CAE/CAE_testlabel.npy')
print("basic information of CAE dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 20, 
                          enhencetimes = 20,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 50, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()

print('/*===================HTR dataset=========================*/')
traindata = np.load(path+'/Data/HTR/HTR_traindata.npy')
trainlabel = np.load(path+'/Data/HTR/HTR_trainlabel.npy')
testdata = np.load(path+'/Data/HTR/HTR_testdata.npy')
testlabel = np.load(path+'/Data/HTR/HTR_testlabel.npy')
print("basic information of HTR dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 20, 
                          enhencetimes = 20,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 50, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()

print('/*===================MUS dataset=========================*/')
traindata = np.load(path+'/Data/MUS/MUS_traindata.npy')
trainlabel = np.load(path+'/Data/MUS/MUS_trainlabel.npy')
testdata = np.load(path+'/Data/MUS/MUS_testdata.npy')
testlabel = np.load(path+'/Data/MUS/MUS_testlabel.npy')
print("basic information of MUS dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()

gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 10, 
                          enhencetimes = 10,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 100, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()

print('/*===================CRS dataset=========================*/')
traindata = np.load(path+'/Data/CRS/CRS_traindata.npy')
trainlabel = np.load(path+'/Data/CRS/CRS_trainlabel.npy')
testdata = np.load(path+'/Data/CRS/CRS_testdata.npy')
testlabel = np.load(path+'/Data/CRS/CRS_testlabel.npy')
print("basic information of CRS dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()
gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes = 20, 
                          enhencetimes = 20,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 50, 
                          reg = 1)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS on testdata is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()

print('/*===================BAS dataset=========================*/')
traindata = np.load(path+'/Data/BAS/BAS_traindata.npy')
trainlabel = np.load(path+'/Data/BAS/BAS_trainlabel.npy')
testdata = np.load(path+'/Data/BAS/BAS_testdata.npy')
testlabel = np.load(path+'/Data/BAS/BAS_testlabel.npy')
print("basic information of BAS dataset:")
print("the traindata shape : ", traindata.shape)
print("the testdata shape  : ", testdata.shape)
print()
print("BNGBS begins training...")
print()
gbn = GradientBoostingNet(column_sampling = 1, 
                          row_sampling = 1, 
                          learning_rate = 0.1,
                          n_estimators = 10,
                          maptimes =50, 
                          enhencetimes = 50,
                          map_function = 'tanh',
                          enhence_function = 'tanh',
                          batchsize = 20, 
                          reg = 2)

gbn.fit(traindata,trainlabel)
predictlabel = gbn.predict(testdata)
print('the accuracy of BNGBS is {0}'.format(show_accuracy(predictlabel,testlabel)))
print()

