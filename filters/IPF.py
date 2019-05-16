# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:18:24 2019

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
def IPF(b,votacion,folds,ruido):
    DaTSe = 0
    for DataSet in b:
        bags,labels,X = load_data(DataSet)
        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
        skf = StratifiedKFold(n_splits=folds)
#        dataAcc = np.zeros((len(b),len(ruido),folds,2))
        print('\n\tDATASET: '+str(DataSet)+'\n')
        for ny,k in enumerate(ruido):
            print('\t\t=>RUIDO : '+str(k))
            fold = 1
            results_Fil = []
            results_Ori = []
            for train_index, test_index in skf.split(bags, labels.reshape(len(labels))):
                print('\t\t  =>FOLD : '+str(fold))
                X_train = [bags[i] for i in train_index]        
                Y_train = labels[train_index]
                X_test  = [bags[i] for i in test_index]
                Y_test  = labels[test_index]
                X_train_NoNy,Y_train_NoNy = fun_aux.mil_cv_filter_ef(X_train,Y_train,folds,votacion)
                print('\t\t\t=>Original')
                results_Ori.append(fun_aux.filtrado_final(X_train,Y_train,X_test,Y_test)) 
                print('\t\t\t=>Filtrado')
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


def IPF(b,votacion,folds,ruido):
    error = 0.01
    toStop = 3
    for DataSet in b:
        bags,labels,X = load_data(DataSet)
        #bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
        Clasificadores = fun_aux.clasif()
    
#        isCorrectLabel = np.ones((len(Clasificadores), len(labels)), dtype=bool)
        print('\n\tDATASET: '+str(DataSet)+'\n')
        for ny,k in enumerate(ruido):
            print('\t\t=>RUIDO : '+str(k))
            for i,cl in enumerate(Clasificadores):
                
                stop = True
                load = True
                countToStop = 0
                vuelta = 0
                Cop_bags = copy.copy(bags)
                Cop_labe = copy.copy(labels)
                while stop:
                    skf = StratifiedKFold(n_splits=folds)
#                    print('vuelta : '+str(vuelta))
                    if load:
                        
                        isCorrectLabel = np.ones((folds, len(Cop_labe)), dtype=bool)
                        for j in range(1,folds+1):
#                            print('\n********** Fold: '+str(j)+' **********\n')
                            train_index,Y_train,test_index,Y_test = fun_aux.loadNoisy(DataSet,k,j)
                            X_train = [Cop_bags[g] for g in train_index]     
                            X_test  = [Cop_bags[g] for g in test_index]
                            bag_all = np.concatenate((X_train, X_test), axis=None)
                            label_all = np.concatenate((Y_train, Y_test), axis=None)
                           
                            try:
                                if len(Clasificadores[i][1]) > 0:
                                    Clasificadores[i][0].fit(X_train, Y_train, **Clasificadores[i][1])
                                else:
                                    Clasificadores[i][0].fit(X_train, Y_train)
                                predictions = Clasificadores[i][0].predict(X_test) 
                                if (isinstance(predictions, tuple)):
                                    predictions = predictions[0]
                            except:
                                print('Fallo, segundo intento')
                                try:
                                    if len(Clasificadores[i][1]) > 0:
                                        Clasificadores[i][0].fit(X_train, Y_train, **Clasificadores[i][1])
                                    else:
                                        Clasificadores[i][0].fit(X_train, Y_train)
                                    predictions = Clasificadores[i][0].predict(X_test) 
                                    if (isinstance(predictions, tuple)):
                                        predictions = predictions[0]
                                    print('OK')
                                except:
                                    print('Fallo en calculo en carga')

                            for l,p in enumerate(test_index):    
                                isCorrectLabel[j-1][p] = (Y_test[l] == np.sign(predictions[l]))
                    else:
                        j = 1
                        isCorrectLabel = np.ones((folds, len(label_all)), dtype=bool)
                        for train_index, test_index in skf.split(bag_all, label_all.reshape(len(label_all))):
#                            print('\n********** Fold: '+str(j)+' **********\n')
                            
                            X_train = [bag_all[g] for g in train_index]        
                            Y_train = label_all[train_index]
                            X_test  = [bag_all[g] for g in test_index]
                            Y_test  = label_all[test_index]
                            intento_shuff = 0                            
                            while (np.sum(Y_train) in [len(Y_train),0]) and intento_shuff < 4:
#                                print('Solo hay una clase')                                
                                all_index = np.concatenate((train_index, test_index))
                                ToChange = fun_aux.Porcentaje(len(all_index),20)
#                                ToChangeResto = len(all_index) - ToChange                  
                                pru1 = np.split(all_index,[ToChange,len(all_index)])
                                train_index = pru1[0]
                                test_index = pru1[1]
#                                train_index = rand.sample(range(0,len(all_index)-1),ToChange)
#                                test_index = rand.sample(range(0,len(all_index)-1),ToChangeResto)
                                X_train = [bag_all[g] for g in train_index]        
                                Y_train = label_all[train_index]
                                X_test  = [bag_all[g] for g in test_index]
                                Y_test  = label_all[test_index]
                                if intento_shuff == 3:
                                    if Y_train[0] == 0:
                                        Y_train[0] = 1
                                    else:
                                        Y_train[0] = 0
#                                print('vuelta'+str(intento_shuff))
                                intento_shuff = intento_shuff + 1
                                
                            try:
                                if len(Clasificadores[i][1]) > 0:
                                    Clasificadores[i][0].fit(X_train, Y_train, **Clasificadores[i][1])
                                else:
                                    Clasificadores[i][0].fit(X_train, Y_train)
                                predictions = Clasificadores[i][0].predict(X_test) 
                                if (isinstance(predictions, tuple)):
                                    predictions = predictions[0]
                                
                            except:
                                print('Fallo, segundo intento')
                                try:
                                    if len(Clasificadores[i][1]) > 0:
                                        Clasificadores[i][0].fit(X_train, Y_train, **Clasificadores[i][1])
                                    else:
                                        Clasificadores[i][0].fit(X_train, Y_train)
                                    predictions = Clasificadores[i][0].predict(X_test) 
                                    if (isinstance(predictions, tuple)):
                                        predictions = predictions[0]
                                    print('Ok')
                                except:
                                    print('Fallo al calcular en la reduccion')
                                
                            for l,p in enumerate(test_index):    
                                isCorrectLabel[j-1][p] = (Y_test[l] == np.sign(predictions[l]))
                            j = j + 1
                    if votacion == 'maxVotos':
                        noisyBags = [] 
                        for n in range(0,len(label_all)):
                            aux = 0
                            for m in range(0,folds):
                                if not isCorrectLabel[m][n]:
                                    aux = aux+1
                            if aux > folds/2:
                                noisyBags.append(n)
                    if votacion == 'consenso':
                        noisyBags = []
                        for n in range(0,len(label_all)):
                            aux = True
                            for m in range(0,folds):
                                if aux:
                                    if isCorrectLabel[m][n]:
                                        aux = False
                            if aux:
                                noisyBags.append(n)

                    nonNoisyBags = crearDataSet_noisy(noisyBags,bag_all,labels,DataSet,folds)
                    if len(noisyBags) < (len(bag_all)*error):
                        countToStop = countToStop + 1
                    else:
                        countToStop = 0
                    if countToStop == toStop:
                        stop = False
                    else:
                        load = False
                        bag_all = [bag_all[d] for d in nonNoisyBags] 
                        label_all = label_all[nonNoisyBags]
                        bag_all,label_all = shuffle(bag_all, label_all, random_state=rand.randint(0, 100))

                    vuelta = vuelta + 1
                crearDataSet(noisyBags,bag_all,label_all,DataSet,folds,i,k,Cop_bags,Cop_labe)
def crearDataSet_noisy(noisyBags,bags,labels,DataSet,folds):
    nonNoisyBags = []
    cont = 0
    if len(noisyBags) == 0:
        for z in range(0,len(bags)):
            nonNoisyBags.append(z)
    else:
        for z in range(0,len(bags)):
            if not z == noisyBags[cont]:
                nonNoisyBags.append(z)
            else:
                if cont < len(noisyBags)-1:
                    cont = cont+1
    
    return nonNoisyBags               
    
def crearDataSet(noisyBags,bag_all,label_all,DataSet,folds,i,k,Cop_bags,Cop_labe):
    nonNoisyBags = []
    cont = 0
    if len(noisyBags) == 0:
        for z in range(0,len(bag_all)):
            nonNoisyBags.append(z)
    else:
        for z in range(0,len(bag_all)):
            if not z == noisyBags[cont]:
                nonNoisyBags.append(z)
            else:
                if cont < len(noisyBags)-1:
                    cont = cont+1
    mil_cross_noisy(nonNoisyBags,DataSet,bag_all,label_all,folds,i,k,Cop_bags,Cop_labe)
def mil_cross_noisy(nonNoisyBags,DataSet,bag_all,label_all,folds,i,k,Cop_bags,Cop_labe):  
    Clasificadores = fun_aux.clasif()
    clasific_ny = Clasificadores[i]
    bags_noNoisy = [bag_all[d] for d in nonNoisyBags] 
    labels_noNoisy = label_all[nonNoisyBags]

#    bags_noNoisy,labels_noNoisy = shuffle(bags_noNoisy, labels_noNoisy, random_state=rand.randint(0, 100))
    skf = StratifiedKFold(n_splits=folds)
    

    
    
    
    print('\t\t\t-->Clasificador :'+str(clasific_ny[2]))
    results_accuracie = []
    results_auc = []
    j = 1
    for train_index, test_index in skf.split(bags_noNoisy, labels_noNoisy.reshape(len(labels_noNoisy))):
        #-----------------------------
        train_index_B,Y_train_B,test_index_B,Y_test_B = fun_aux.loadNoisy(DataSet,k,j)
        X_train_B = [Cop_bags[g_B] for g_B in train_index_B]     
        X_test_B  = [Cop_bags[g_B] for g_B in test_index_B]
        bag_all_B = np.concatenate((X_train_B, X_test_B), axis=None)
        label_all_B = np.concatenate((Y_train_B, Y_test_B), axis=None)
        #-----------------------
        
        X_train = [bags_noNoisy[x] for x in train_index]        
        Y_train = labels_noNoisy[train_index]
        X_test  = [bags_noNoisy[x] for x in test_index]
        Y_test  = labels_noNoisy[test_index]
        intento_shuff = 0                            
        while (np.sum(Y_train) in [len(Y_train),0]) and intento_shuff < 4:
#                                print('Solo hay una clase')                                
            all_index = np.concatenate((train_index, test_index))
            ToChange = fun_aux.Porcentaje(len(all_index),20)
#                                ToChangeResto = len(all_index) - ToChange                  
            pru1 = np.split(all_index,[ToChange,len(all_index)])
            train_index = pru1[0]
            test_index = pru1[1]
#                                train_index = rand.sample(range(0,len(all_index)-1),ToChange)
#                                test_index = rand.sample(range(0,len(all_index)-1),ToChangeResto)
            X_train = [bag_all[g] for g in train_index]        
            Y_train = label_all[train_index]
            X_test  = [bag_all[g] for g in test_index]
            Y_test  = label_all[test_index]
            if intento_shuff == 3:
                if Y_train[0] == 0:
                    Y_train[0] = 1
                else:
                    Y_train[0] = 0
#                                print('vuelta'+str(intento_shuff))
            intento_shuff = intento_shuff + 1
        try:
            if len(clasific_ny[1]) > 0:
                clasific_ny[0].fit(X_train, Y_train, **clasific_ny[1])
            else:
                clasific_ny[0].fit(X_train, Y_train)
            predictions = clasific_ny[0].predict(X_test_B) 
            if (isinstance(predictions, tuple)):
                predictions = predictions[0]
            accuracie = np.average(Y_test_B.T == np.sign(predictions)) 
            results_accuracie.append(100 * accuracie)
            auc_score = fun_aux.roc_auc_score_FIXED(Y_test_B,predictions)  
            results_auc.append(100 * auc_score)
            
        except:
            print('Fallo, segundo intento')
          
            try:
                if len(clasific_ny[1]) > 0:
                    clasific_ny[0].fit(X_train, Y_train, **clasific_ny[1])
                else:
                    clasific_ny[0].fit(X_train, Y_train)
                predictions = clasific_ny[0].predict(X_test_B) 
                if (isinstance(predictions, tuple)):
                    predictions = predictions[0]
                accuracie = np.average(Y_test_B.T == np.sign(predictions)) 
                results_accuracie.append(100 * accuracie)
                auc_score = fun_aux.roc_auc_score_FIXED(Y_test_B,predictions)  
                results_auc.append(100 * auc_score)
               
                print('OK-')
               
            except:
                print('Fallo al calcular')
        j = j + 1

             
    print('\t\t\t\t Precisión Media: '+ str(np.mean(results_accuracie))+'%\n\t\t\t\t Media Roc Score: '+ str(np.mean(results_auc)))
    
