import os
import pickle
import sys
import gzip
import numpy as np
from  sknn.mlp import Classifier, Convolution, Layer
analysis_data_path = '../data/analysis/bmi_analysis.pk1'
train_data_path = '../data/train_data/pose_recog_train_data.pk1'
train_data_paths = ['../data/train_data/pose_recog_train_data_0.pk1',
                    '../data/train_data/pose_recog_train_data_1.pk1',
                    '../data/train_data/pose_recog_train_data_2.pk1',
                    '../data/train_data/pose_recog_train_data_3.pk1']

def covnetTrain(train_bmi , train_labels , ite =10 , kernel =3 ,learn_rate =0.02, channel = 8):
    nn = Classifier(
        layers = [
            Convolution("Rectifier", channels=channel, kernel_shape=(kernel,kernel)),
            Layer("Softmax")],
        learning_rate=learn_rate,
        n_iter=ite
        )

    neuralnet = nn.fit(train_bmi , train_labels)
    return  neuralnet

def get_train_test_data(num_test = 0 , function =2):
    test_bmi = []
    test_labels = []
    data_path = train_data_paths[function]
    #get train data
    if os.path.exists(data_path):
        with open(data_path , 'rb') as inp:
            [train_bmi,train_labels] = pickle.load(inp)

    # now segregate test data from train data (since we dont have separate test data set)
    #we have to get test data set somewhere
    label_set = set(train_labels)
    for label in label_set:
        if train_labels.count(label) <= num_test:
            print('number of train instances of ' + label + ' is less than required instances')
        else:
            # extract num_test instances from train data set
            #print('making test instances for '+label)
            
            for i in range(num_test):
                index = train_labels.index(label)
                test_bmi.append(train_bmi[index])
                test_labels.append(train_labels[index])
                del train_bmi[index]
                del train_labels[index]

    return [np.asarray(train_bmi) , np.asarray(train_labels) , np.asarray(test_bmi) , np.asarray(test_labels) ]
            
            
def BMI_covnet_test():
    analysis_input_list = []
    accuracy_list = []
    
    bmi_weight_function = [0,1,2,3]
    for ite in range(3,12):
        for rate in [0.01 , 0.02 , 0.03 , 0.04 , 0.05 , 0.06 ,0.07 , 0.08 , 0.09 , 0.1]:
                
            train_bmi , train_labels , test_bmi , test_labels = get_train_test_data(1 )
            neuralnet = covnetTrain(train_bmi , train_labels , ite ,learn_rate = rate )
            test_y = neuralnet.predict(test_bmi)
            test_y = list(test_y)
            test_y = [a[0] for a in test_y]
            test_labels = list(test_labels)
            accuracy = 0
            for i in range(len(test_y)):
                 if test_y[i] == test_labels[i]:
                     accuracy += 1
                
            accuracy = accuracy / len(test_y)
            analysis_input_list .append([ite ,rate])
            accuracy_list.append(accuracy)

    
    with open(analysis_data_path , 'wb') as out:
        pickle.dump([analysis_input_list,accuracy_list] , out , pickle.HIGHEST_PROTOCOL)
    
                        
                        
def save(nn):
    with open('../data/train_data/CNN.pk1' , 'wb') as out:
        pickle.dump(nn , out , pickle.HIGHEST_PROTOCOL)
        
def getTrainedCNN():
    train_bmi , train_labels ,test_bmi ,test_labels = get_train_test_data(1)
    nn = covnetTrain(train_bmi , train_labels)
    

def main():
    BMI_covnet_test()

if __name__ == '__main__':
    main()

