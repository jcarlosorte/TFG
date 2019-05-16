# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:40:14 2019

@author: Juan Carlos
"""

#imports
import warnings
import numpy as np
from data import load_data
warnings.filterwarnings('ignore')

from funciones import fun_aux

def CVcF(b,votacion,folds,ruido):
    for DataSet in b:
        bags,labels,X = load_data(DataSet)
        #bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
        Clasificadores = fun_aux.clasif()
    
        isCorrectLabel = np.ones((len(Clasificadores), len(labels)), dtype=bool)
        print('\n\tDATASET: '+str(DataSet)+'\n')
        for ny,k in enumerate(ruido):
            print('\t\t=>RUIDO : '+str(k))
            for j in range(1,folds+1):
                train_index,Y_train,test_index,Y_test = fun_aux.loadNoisy(DataSet,k,j)
                X_train = [bags[i] for i in train_index]     
                X_test  = [bags[i] for i in test_index]
                bag_all = np.concatenate((X_train, X_test), axis=None)
                label_all = np.concatenate((Y_train, Y_test), axis=None)
                for i,cl in enumerate(Clasificadores):
                    try:
                        if len(Clasificadores[i][1]) > 0:
                            Clasificadores[i][0].fit(X_train, Y_train, **Clasificadores[i][1])
                        else:
                            Clasificadores[i][0].fit(X_train, Y_train)
                        predictions = Clasificadores[i][0].predict(bag_all) 
                        if (isinstance(predictions, tuple)):
                            predictions = predictions[0]
                    except:
                        print('Fallo, segundo intento')
                        try:
                            if len(Clasificadores[i][1]) > 0:
                                Clasificadores[i][0].fit(X_train, Y_train, **Clasificadores[i][1])
                            else:
                                Clasificadores[i][0].fit(X_train, Y_train)
                            predictions = Clasificadores[i][0].predict(bag_all) 
                            if (isinstance(predictions, tuple)):
                                predictions = predictions[0]
                            print('OK')
                        except:
                            print('Fallo en calculo')
                   
#                    for l,p in enumerate(test_index):    
#                        isCorrectLabel[i][p] = (Y_test.T[0][l] == np.sign(predictions[l]))
                    for l in range(0,len(label_all)):    
                        isCorrectLabel[i][l] = (label_all[l] == np.sign(predictions[l]))

        #        print('========= Fold :'+str(j)+' =========')
            
            if votacion == 'maxVotos':
                noisyBags = []
                for n in range(0,len(labels)):
                    aux = 0
                    for m in range(0,len(Clasificadores)):
                        if not isCorrectLabel[m][n]:
                            aux = aux+1
                    if aux > len(Clasificadores)/2:
                        noisyBags.append(n)
            if votacion == 'consenso':
                noisyBags = []
                for n in range(0,len(labels)):
                    aux = True
                    for m in range(0,len(Clasificadores)):
                        if aux:
                            if isCorrectLabel[m][n]:
                                aux = False
                    if aux:
                        noisyBags.append(n)
        #    print('Noisy instances = ')
        #    print(noisyBags)
        #    print('Total = '+str(len(noisyBags)))
            fun_aux.crearDataSet(noisyBags,bags,labels,DataSet,folds)

    