# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:45:16 2019

@author: Juan Carlos
"""

import sys,os,csv,shutil
import warnings
os.chdir('C:/Users/Administrador/Documents/GitHub/TFG/MILpy')
sys.path.append(os.path.realpath('..'))
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random as rand
import numpy as np
from data import load_data
warnings.filterwarnings('ignore')
from MILpy.functions.mil_cross_val import mil_cross_val

#Import Algorithms 
from MILpy.Algorithms.simpleMIL import simpleMIL
from MILpy.Algorithms.MILBoost import MILBoost
from MILpy.Algorithms.maxDD import maxDD
from MILpy.Algorithms.CKNN import CKNN
from MILpy.Algorithms.EMDD import EMDD
#from MILpy.Algorithms.MILES import MILES
from MILpy.Algorithms.BOW import BOW

def Porcentaje(X,Y):
    return X*Y/100

folds = 5
runs = 1
DataSet = ['Fox_scaled']#pruebas
SMILaMax = [simpleMIL(),{'type': 'max'},'MIL max']
SMILaMin = [simpleMIL(),{'type': 'min'},'MIL min']
Clasificadores = [SMILaMax,SMILaMin]
Parametros = [{'type': 'max'},{'type': 'min'}]
#DataSet = ['musk1_scaled','Musk2_scaled','Elephant_scaled','Fox_scaled','mutagenesis1_scaled','mutagenesis2_scaled','Tiger_scaled']
carpeta = '../dataNoisy/'
filename1 = 'X_train_bags.csv'
filename2 = 'Y_train_labels.csv'
filename3 = 'X_test_bags.csv'
filename4 = 'Y_test_labels.csv'

NoisyPercent = [0,5,10,15,20,25,30]

for j in DataSet:
    fold = 1
    results_accuracie = []
    results_auc = []
    print '\n********** DATASET: ',j,' **********\n'
    bags,labels,X = load_data(j)
    bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
    try:
        shutil.rmtree(carpeta+j)
    except:
        print('Se creará la carpeta para el dataset')
    skf = StratifiedKFold(labels.reshape(len(labels)), n_folds=folds)
    
    for train_index, test_index in skf:
        print('========= Fold :'+str(fold)+' =========')
        X_train = [bags[i] for i in train_index]        
        Y_train = labels[train_index]
        X_test  = [bags[i] for i in test_index]
        Y_test  = labels[test_index]
        for k in NoisyPercent:
            if k == 0:
                carpetaSub = carpeta+j+'/fold_'+str(fold)+'/Original/'
            else:
                carpetaSub = carpeta+j+'/fold_'+str(fold)+'/Noisy_'+str(k)+'/'

            LabelToChange = Porcentaje(len(train_index),k)
            aleatorios = rand.sample(range(0,len(train_index)-1),LabelToChange)
            for al in aleatorios:
                if Y_train[al] == 0:
                    Y_train[al] = Y_train[al]+1
                else:
                    Y_train[al] = Y_train[al]-1
#            print('-> Noisy :'+str(k))
            #============================================
            tp = 0
            for cl in Clasificadores:
#                print '\n========= CLASIFICADOR: ',str(cl[2]),' ========='
                if len(cl[1]) > 0:
                    cl[0].fit(X_train, Y_train, **cl[1])
                else:
                    cl[0].fit(bags, labels)
                predictions = cl[0].predict(X_test) 
                if (isinstance(predictions, tuple)):
                    predictions = predictions[0]
                accuracie = (100 * np.average(Y_test.T == np.sign(predictions)))
                
                auc_score = (100 * roc_auc_score(Y_test,predictions))  
                
#                print '\n Precisión: '+ str(auc_score)
                print('Clasificador :'+str(cl[2])+'\n\t Noisy :'+str(k)+'\n\t Precisión: '+ str(auc_score))
                tp = tp+1
            #============================================
            try:
                os.stat(carpetaSub)
            except:
                os.makedirs(carpetaSub)
            with open(carpetaSub+filename1, 'wb') as csvfile:
                writer = csv.writer(csvfile)
                for i in train_index:
                    writer.writerow([i])
            with open(carpetaSub+filename2, 'wb') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(Y_train)
            with open(carpetaSub+filename3, 'wb') as csvfile:
                writer = csv.writer(csvfile)
                for i in test_index:
                    writer.writerow([i])
            with open(carpetaSub+filename4, 'wb') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(Y_test)
        fold = fold+1