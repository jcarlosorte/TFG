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
from sklearn.metrics import roc_auc_score, accuracy_score
from funciones import fun_aux

def EF(b,votacion,folds,ruido):
    DaTSe = 0
    for DataSet in b:
        bags,labels,X = load_data(DataSet)
        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, len(labels)-1))
        skf = StratifiedKFold(n_splits=folds)
#        dataAcc = np.zeros((len(b),len(ruido),folds,2))
        print('\n\tDATASET: '+str(DataSet)+'\n')
        for ny,k in enumerate(ruido):
            print('\t\t=>RUIDO : '+str(k))
            fold = 1
            results_Fil = []
            results_Ori = []
            for train_index, test_index in skf.split(bags, labels.reshape(len(labels))):
                print('\t\t\t=>FOLD : '+str(fold))
                X_train = [bags[i] for i in train_index]        
                Y_train = labels[train_index]
                X_test  = [bags[i] for i in test_index]
                Y_test  = labels[test_index]
                LabelToChange = fun_aux.Porcentaje(len(train_index),k)
                aleatorios = rand.sample(range(0,len(train_index)),int(LabelToChange))
                for al in aleatorios:
                    if Y_train[al] == 0:
                        Y_train[al] = Y_train[al]+1
                    else:
                        Y_train[al] = Y_train[al]-1
                X_train_NoNy,Y_train_NoNy = fun_aux.mil_cv_filter_ef(X_train,Y_train,folds,votacion)
#                print('\t\t\t=>Original')
                results_Ori.append(fun_aux.filtrado_final(X_train,Y_train,X_test,Y_test)) 
#                print('\t\t\t=>Filtrado')
                results_Fil.append(fun_aux.filtrado_final(X_train_NoNy,Y_train_NoNy,X_test,Y_test))
                fold = fold + 1
            Clasificadores = fun_aux.clasif()
            results_accuracie_F = []
            results_auc_F = []
            results_accuracie_O = []
            results_auc_O = []
            for h in range(0,len(Clasificadores)):
                print('\t\t\t\t-->Clasificador :'+str(Clasificadores[h][2]))
                for g in range(0,folds):
                    results_accuracie_F.append(results_Fil[g][h][0])
                    results_auc_F.append(results_Fil[g][h][1])
                    results_accuracie_O.append(results_Ori[g][h][0])
                    results_auc_O.append(results_Ori[g][h][1])
                print('\t\t\t\t\t-->Original')
                print('\t\t\t\t\t Precisión: '+ str(np.mean(results_accuracie_O))+'%')
                print('\t\t\t\t\t Roc Score: '+ str(np.mean(results_auc_O)))
                print('\t\t\t\t\t-->Filtrado')
                print('\t\t\t\t\t Precisión: '+ str(np.mean(results_accuracie_F))+'%')
                print('\t\t\t\t\t Roc Score: '+ str(np.mean(results_auc_F)))
#            dataAcc[DaTSe][ny][fold-1][0] = 
#            dataAcc[DaTSe][ny][fold-1][1] =
    DaTSe = DaTSe + 1

def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1 or len(np.unique(y_true)) == 0: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred) 