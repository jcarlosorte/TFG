# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:40:14 2019

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

def CVcF(b,votacion,folds,ruido):
    DaTSe = 0
    for DataSet in b:
        bags,labels,X = load_data(DataSet)
        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
        skf = StratifiedKFold(n_splits=folds)
#        dataAcc = np.zeros((len(b),len(ruido),folds,2))
        print('\n\tDATASET: '+str(DataSet)+'\n')
        
        for ny,k in enumerate(ruido):
            print('\t\t=>RUIDO : '+str(k))
            Clasificadores = fun_aux.clasif()
            Clasificadores_filtro = fun_aux.cla_filter_cvcf()
            for s,cl in enumerate(Clasificadores):
                fold = 1
#                results_Fil = np.zeros((len(Clasificadores_filtro),folds))
                results_Fil = [[] for x in range(len(Clasificadores_filtro))] 
                results_Ori = []
                clasificador_ = Clasificadores[s]
                for train_index, test_index in skf.split(bags, labels.reshape(len(labels))):
                    print('\t\t  =>FOLD : '+str(fold))
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

                    for j,cl_f in enumerate(Clasificadores_filtro):
                        clasificador_f = Clasificadores_filtro[j]
#                        print('\t\t\t=>Filtrado con '+str(cl_f[2]))
                        X_train_NoNy,Y_train_NoNy = mil_cv_filter_cvcf(X_train,Y_train,folds,votacion,clasificador_f) 
                        results_Fil[j].append(filtrado_final(X_train_NoNy,Y_train_NoNy,X_test,Y_test,clasificador_))
#                    print(len(X_train_NoNy))
#                    print(len(X_train))
#                    print('\t\t\t=>Original')
                    results_Ori.append(filtrado_final(X_train,Y_train,X_test,Y_test,clasificador_))
                    fold = fold + 1
                results_accuracie_O = []
                results_auc_O = []
                print('\t\t\t\t-->Clasificador :'+str(clasificador_[2]))
                for g in range(0,folds):
                    results_accuracie_O.append(results_Ori[g][0])
                    results_auc_O.append(results_Ori[g][1])
                print('\t\t\t\t\t-->Original')
                print('\t\t\t\t\t Precisión: '+ str(np.mean(results_accuracie_O, dtype=np.float64))+'%')
                print('\t\t\t\t\t Roc Score: '+ str(np.mean(results_auc_O, dtype=np.float64)))
                
                for h,cl_f0 in enumerate(Clasificadores_filtro):
                    results_accuracie_F = []
                    results_auc_F = []
                    print('\t\t\t\t\t-->Filtrado por '+str(cl_f0[2]))
                    for g in range(0,folds):
                        results_accuracie_F.append(results_Fil[j][g][0])
                        results_auc_F.append(results_Fil[j][g][1])  
                    print('\t\t\t\t\t Precisión: '+ str(np.mean(results_accuracie_F, dtype=np.float64))+'%')
                    print('\t\t\t\t\t Roc Score: '+ str(np.mean(results_auc_F, dtype=np.float64)))
#            dataAcc[DaTSe][ny][fold-1][0] = 
#            dataAcc[DaTSe][ny][fold-1][1] =
    DaTSe = DaTSe + 1

def mil_cv_filter_cvcf(bags_f,labels_f,folds,votacion,clasificador_):
#    print('\t\t\tFiltrando...')
    bags_f,labels_f = shuffle(bags_f, labels_f, random_state=rand.randint(0, 100))
    skf = StratifiedKFold(n_splits=folds)
    isCorrectLabel = np.ones((folds, len(labels_f)), dtype=bool)
    fold = 0
    for train_index, test_index in skf.split(bags_f, labels_f.reshape(len(labels_f))):
        X_train = [bags_f[i] for i in train_index]        
        Y_train = labels_f[train_index]
        X_test  = [bags_f[i] for i in test_index]
        Y_test  = labels_f[test_index]
        try:
            if len(clasificador_[1]) > 0:
                clasificador_[0].fit(X_train, Y_train, **clasificador_[1])
            else:
                clasificador_[0].fit(X_train, Y_train)
            predictions = clasificador_[0].predict(X_train)
            if (isinstance(predictions, tuple)):
                predictions = predictions[0]
        except:
            print('Fallo, segundo intento')
            try:
                if len(clasificador_[1]) > 0:
                    clasificador_[0].fit(X_train, Y_train, **clasificador_[1])
                else:
                    clasificador_[0].fit(X_train, Y_train)
                predictions = clasificador_[0].predict(X_train)
                if (isinstance(predictions, tuple)):
                    predictions = predictions[0]
                print('OK')
            except:
                print('Fallo en calculo')
        for l,p in enumerate(train_index): 
            isCorrectLabel[fold][p] = (Y_train.T[0][l] == np.sign(predictions[l]))
        fold = fold + 1
    if votacion == 'maxVotos':
        noisyBags = []
        for n in range(0,len(labels_f)):
            aux = 0
            for m in range(0,folds):
                if not isCorrectLabel[m][n]:
                    aux = aux+1
            if aux > folds/2:
                noisyBags.append(n)
    if votacion == 'consenso':
        noisyBags = []
        for n in range(0,len(labels_f)):
            aux = True
            for m in range(0,folds):
                if aux:
                    if isCorrectLabel[m][n]:
                        aux = False
            if aux:
                print('holi')
                noisyBags.append(n)
    nonNoisyBags = [] 
    cont = 0
    if len(noisyBags) == 0:
        for z in range(0,len(bags_f)):
            nonNoisyBags.append(z)
    else:
        for z in range(0,len(bags_f)):
            if cont < len(noisyBags) and noisyBags[cont] == z:
                cont = cont + 1
            else:
                nonNoisyBags.append(z)
    X_train_NoNy = [bags_f[i] for i in nonNoisyBags]
    Y_train_NoNy = labels_f[nonNoisyBags]
    return X_train_NoNy,Y_train_NoNy

def filtrado_final(X_train,Y_train,X_test,Y_test,clasificador_):  
    results = np.zeros((2))
    accuracie = 0
    auc_score = 0
    try:
        if len(clasificador_[1]) > 0:
            clasificador_[0].fit(X_train, Y_train, **clasificador_[1])
        else:
            clasificador_[0].fit(X_train, Y_train)
        predictions = clasificador_[0].predict(X_test) 
        if (isinstance(predictions, tuple)):
            predictions = predictions[0]
        accuracie = (100 * np.average(Y_test.T == np.sign(predictions))) 
        auc_score = (100 * fun_aux.roc_auc_score_FIXED(Y_test,predictions))
    except:
        print('Fallo, segundo intento')
      
        try:
            if len(clasificador_[1]) > 0:
                clasificador_[0].fit(X_train, Y_train, **clasificador_[1])
            else:
                clasificador_[0].fit(X_train, Y_train)
            predictions = clasificador_[0].predict(X_test) 
            if (isinstance(predictions, tuple)):
                predictions = predictions[0]
            accuracie = (100 * np.average(Y_test.T == np.sign(predictions)))   
            auc_score = (100 * fun_aux.roc_auc_score_FIXED(Y_test,predictions))
            print('OK')     
        except:
            print('Fallo en calculo')      
    results[0] = accuracie
    results[1] = auc_score
#    print('\t\t\t\t\t Precisión: '+ str(accuracie)+'%\n\t\t\t\t\t Roc Score: '+ str(auc_score))
    return results