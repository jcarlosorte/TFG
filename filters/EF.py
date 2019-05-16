# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:37:39 2019

@author: Juan Carlos
"""

#imports
import warnings
from data import load_data
import random as rand
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

from funciones import fun_aux

def EF(b,votacion,folds,ruido):
    for DataSet in b:
        bags,labels,X = load_data(DataSet)
        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
        skf = StratifiedKFold(n_splits=folds)
        
        print('\n\tDATASET: '+str(DataSet)+'\n')
        for ny,k in enumerate(ruido):
            print('\t\t=>RUIDO : '+str(k))
            fold = 1
            for train_index, test_index in skf.split(bags, labels.reshape(len(labels))):
                print('\t\t=>FOLD : '+str(fold))
                X_train = [bags[i] for i in train_index]        
                Y_train = labels[train_index]
                X_test  = [bags[i] for i in test_index]
                Y_test  = labels[test_index]
                
                X_train_NoNy,Y_train_NoNy = fun_aux.mil_cv_filter(X_train,Y_train,folds,votacion)
               
                fun_aux.filtrado_final(X_train,Y_train,X_test,Y_test)

                fun_aux.filtrado_final(X_train_NoNy,Y_train_NoNy,X_test,Y_test)
                
                fold = fold + 1

    