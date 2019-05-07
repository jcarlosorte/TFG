# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:45:16 2019

@author: Juan Carlos
"""
#imports
import sys,os,csv,shutil,copy,warnings
import random as rand
import numpy as np
os.chdir('C:/Users/Administrador/Documents/GitHub/TFG/MILpy')
sys.path.append(os.path.realpath('..'))
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from data import load_data
warnings.filterwarnings('ignore')

from funciones import fun_aux
#definiciones
folds = 5
#DataSet = ['Fox_scaled','Musk2_scaled']#pruebas
#DataSet = ['musk1_scaled','Musk2_scaled','Elephant_scaled','Fox_scaled','mutagenesis1_scaled','mutagenesis2_scaled','Tiger_scaled']
DataSet = ['Fox_scaled','Elephant_scaled']
carpeta = '../dataNoisy/'
filename1 = 'X_train_bags.csv'
filename2 = 'Y_train_labels.csv'
filename3 = 'X_test_bags.csv'
filename4 = 'Y_test_labels.csv'
file_test = '../TestNoisy.txt'
file_data = '../DataNoisy.txt'
NoisyPercent = [0,5,10,15,20,25,30]
try:
    os.stat(file_data)
except:
    h = open(file_data, "w+")
    h.close()
h = open(file_data, "a")
for j in DataSet:
    Clasificadores = fun_aux.clasif()
    print('\n********** DATASET: '+str(j)+' **********\n')
    h.write('\n********** DATASET: '+str(j)+' **********\n')
    bags,labels,X = load_data(j)
    bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
    try:
        shutil.rmtree(carpeta+j)
    except:
        print('Se creará la carpeta para el dataset')
    skf = StratifiedKFold(labels.reshape(len(labels)), n_folds=folds)
    fold = 1
    
    for train_index, test_index in skf:
        print('========= Fold :'+str(fold)+' =========')
        h.write('========= Fold :'+str(fold)+' =========\n')
        X_train = [bags[i] for i in train_index]        
        Y_train = labels[train_index]
        X_test  = [bags[i] for i in test_index]
        Y_test  = labels[test_index]
        Cop_bags = copy.copy(bags)
        Cop_labe = copy.copy(labels)

        for ny,k in enumerate(NoisyPercent):
            if k == 0:
                carpetaSub = carpeta+j+'/fold_'+str(fold)+'/Original/'
            else:
                carpetaSub = carpeta+j+'/fold_'+str(fold)+'/Noisy_'+str(k)+'/'

            LabelToChange = fun_aux.Porcentaje(len(train_index),k)
            aleatorios = rand.sample(range(0,len(train_index)-1),LabelToChange)
            cop_LabelToChange = fun_aux.Porcentaje(len(Cop_labe),k)
            cop_aleatorios = rand.sample(range(0,len(Cop_labe)-1),cop_LabelToChange)
            #cambiar etiquetas en train
            for al in aleatorios:
                if Y_train[al] == 0:
                    Y_train[al] = Y_train[al]+1
                else:
                    Y_train[al] = Y_train[al]-1
            
            for al_c in cop_aleatorios:
                if Cop_labe[al_c] == 0:
                    Cop_labe[al_c] = Cop_labe[al]+1
                else:
                    Cop_labe[al_c] = Cop_labe[al]-1
            print('-> Noisy :'+str(k))
            #============================================
            
            for i,cl in enumerate(Clasificadores):
                accuracie = 0
                auc_score = 0
                try:
                    if len(Clasificadores[i][1]) > 0:
                        Clasificadores[i][0].fit(X_train, Y_train, **Clasificadores[i][1])
                    else:
                        Clasificadores[i][0].fit(Cop_bags, Cop_labe)
                    predictions = Clasificadores[i][0].predict(X_test) 
                    if (isinstance(predictions, tuple)):
                        predictions = predictions[0]
                    accu_aux = np.average(Y_test.T == np.sign(predictions))
                    accuracie = (100 * accu_aux)
                    auc_sco_aux = roc_auc_score(Y_test,predictions)
                    auc_score = (100 * auc_sco_aux)
                    Clasificadores[i][3][ny].append(accuracie)
                    Clasificadores[i][4][ny].append(auc_score)
                except:
                    print('Fallo, segundo intento')
                    h.write('Fallo, segundo intento\n')
                    try:
                        if len(Clasificadores[i][1]) > 0:
                            Clasificadores[i][0].fit(X_train, Y_train, **Clasificadores[i][1])
                        else:
                            Clasificadores[i][0].fit(Cop_bags, Cop_labe)
                        predictions = Clasificadores[i][0].predict(X_test) 
                        if (isinstance(predictions, tuple)):
                            predictions = predictions[0]
                        accu_aux = np.average(Y_test.T == np.sign(predictions))
                        accuracie = (100 * accu_aux)
                        auc_sco_aux = roc_auc_score(Y_test,predictions)
                        auc_score = (100 * auc_sco_aux)
                        Clasificadores[i][3][ny].append(accuracie)
                        Clasificadores[i][4][ny].append(auc_score)
                        print('OK')
                        h.write('OK\n')
                    except:
                        Clasificadores[i][3][ny].append(50)
                        Clasificadores[i][4][ny].append(50)
                        print('Fallo en calculo')
                        h.write('Fallo en calculo:\n')
#                print('Clasificador :'+str(cl[2])+'\n\t Ruido '+str(k)+'%\n\t Score: '+ str(auc_score)+'%\n\t Precision: '+ str(accuracie))
                h.write('Clasificador :'+str(cl[2])+'\n\t Ruido '+str(k)+'%\n\t Score: '+ str(auc_score)+'%\n\t Precision: '+ str(accuracie)+'\n')
            
            #============================================
            #Se sobreescribe cada vez que se ejecutan para sustituir las nuevas particiones
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
    try:
        os.stat(file_test)
    except:
        f = open(file_test, "w+")
        f.close()
    f = open(file_test, "a")
    f.write('\n********** DATASET: '+str(j)+' **********\n')
    for h2,clasi in enumerate(Clasificadores):
        print('Clasificador: '+str(clasi[2]))
        f.write('Clasificador: '+str(clasi[2])+'\n')
        for p,noy in enumerate(NoisyPercent):
            print('\t=>Ruido: '+str(noy)+'%\tPrecisión Media: '+str(np.mean(clasi[3][p]))+'%\n\t\t\tMedia Roc Score: '+str(np.mean(clasi[4][p])))
            f.write('\t=>Ruido: '+str(noy)+'%\tPrecisión Media: '+str(np.mean(clasi[3][p]))+'%\n\t\t\tMedia Roc Score: '+str(np.mean(clasi[4][p]))+'\n')
    f.close()
h.close()