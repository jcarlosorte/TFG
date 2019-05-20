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
#from funciones import fun_aux
#Import Algorithms 
from MILpy.Algorithms.simpleMIL import simpleMIL
from MILpy.Algorithms.MILBoost import MILBoost
from MILpy.Algorithms.maxDD import maxDD
from MILpy.Algorithms.CKNN import CKNN
from MILpy.Algorithms.EMDD import EMDD
from MILpy.Algorithms.MILES import MILES
from MILpy.Algorithms.BOW import BOW

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
#                print('\t\t\t=>FOLD : '+str(fold))
                X_train = [bags[i] for i in train_index]        
                Y_train = labels[train_index]
                X_test  = [bags[i] for i in test_index]
                Y_test  = labels[test_index]
                LabelToChange = Porcentaje(len(train_index),k)
                aleatorios = rand.sample(range(0,len(train_index)),int(LabelToChange))
                for al in aleatorios:
                    if Y_train[al] == 0:
                        Y_train[al] = Y_train[al]+1
                    else:
                        Y_train[al] = Y_train[al]-1
                X_train_NoNy,Y_train_NoNy = mil_cv_filter_ef(X_train,Y_train,folds,votacion)
#                print('\t\t\t=>Original')
                results_Ori.append(filtrado_final(X_train,Y_train,X_test,Y_test)) 
#                print('\t\t\t=>Filtrado')
                results_Fil.append(filtrado_final(X_train_NoNy,Y_train_NoNy,X_test,Y_test))
                fold = fold + 1
            Clasificadores = clasif()
            Clasificadores_filtro = cla_filter()
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
                print('\t\t\t\t\t-->Filtrado por ')
                for h,cl_f0 in enumerate(Clasificadores_filtro):
                    print('\t\t\t\t\t * '+str(cl_f0[2]))
#                print('\t\t\t\t\t-->Filtrado')
                print('\t\t\t\t\t Precisión: '+ str(np.mean(results_accuracie_F))+'%')
                print('\t\t\t\t\t Roc Score: '+ str(np.mean(results_auc_F)))
#            dataAcc[DaTSe][ny][fold-1][0] = 
#            dataAcc[DaTSe][ny][fold-1][1] =
    DaTSe = DaTSe + 1

def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1 or len(np.unique(y_true)) == 0: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)

def Porcentaje(X,Y):
    return X*Y/100

def clasif():
    aux = []
    resul1 = [[],[],[],[],[],[],[]]
    resul2 = [[],[],[],[],[],[],[]]
    resul3 = [[],[],[],[],[],[],[]]
    resul4 = [[],[],[],[],[],[],[]]
    resul5 = [[],[],[],[],[],[],[]]
    resul6 = [[],[],[],[],[],[],[]]
    resul7 = [[],[],[],[],[],[],[]]
    resul8 = [[],[],[],[],[],[],[]]
    resul9 = [[],[],[],[],[],[],[]]
    roc_m_1 = [[],[],[],[],[],[],[]]
    roc_m_2 = [[],[],[],[],[],[],[]]
    roc_m_3 = [[],[],[],[],[],[],[]]
    roc_m_4 = [[],[],[],[],[],[],[]]
    roc_m_5 = [[],[],[],[],[],[],[]]
    roc_m_6 = [[],[],[],[],[],[],[]]
    roc_m_7 = [[],[],[],[],[],[],[]]
    roc_m_8 = [[],[],[],[],[],[],[]]
    roc_m_9 = [[],[],[],[],[],[],[]]
    SMILaMax = [simpleMIL(),{'type': 'max'},'MIL max',resul1,roc_m_1]
    SMILaMin = [simpleMIL(),{'type': 'min'},'MIL min',resul2,roc_m_2]
    SMILaExt = [simpleMIL(),{'type': 'extreme'},'MIL Extreme',resul3,roc_m_3]
    BOW_clas = [BOW(),{'k':90,'covar_type':'diag','n_iter':20},'BOW',resul4,roc_m_4]
    CKNN_cla = [CKNN(),{'references': 3, 'citers': 5},'CKNN',resul5,roc_m_5]
    maxDD_cl = [maxDD(),{},'DIVERSE DENSITY',resul6,roc_m_6]
    EMDD_cla = [EMDD(),{},'EM-DD',resul7,roc_m_7]
    MILB_cla = [MILBoost(),{},'MILBOOST',resul8,roc_m_8]
    MILES_cl = [MILES(),{},'MILES',resul9,roc_m_9]
#    aux.append(SMILaMax)
    aux.append(SMILaMin)
#    aux.append(SMILaExt)
    aux.append(BOW_clas)
#    aux.append(CKNN_cla)
#    aux.append(maxDD_cl)
#    aux.append(EMDD_cla)
#    aux.append(MILB_cla)
#    aux.append(MILES_cl)
    return aux

def mil_cv_filter_ef(bags_f,labels_f,folds,votacion):
    print('\t\t\tFiltrando...')
    Clasificadores = cla_filter()
    bags_f,labels_f = shuffle(bags_f, labels_f, random_state=rand.randint(0, 100))
    skf = StratifiedKFold(n_splits=folds)
    isCorrectLabel = np.ones((len(Clasificadores), len(labels_f)), dtype=bool)
    for train_index, test_index in skf.split(bags_f, labels_f.reshape(len(labels_f))):
        X_train = [bags_f[i] for i in train_index]        
        Y_train = labels_f[train_index]
        X_test  = [bags_f[i] for i in test_index]
        Y_test  = labels_f[test_index]
        for s,cl in enumerate(Clasificadores):
            
            try:
                if len(Clasificadores[s][1]) > 0:
                    Clasificadores[s][0].fit(X_train, Y_train, **Clasificadores[s][1])
                else:
                    Clasificadores[s][0].fit(X_train, Y_train)
                predictions = Clasificadores[s][0].predict(X_test)
                if (isinstance(predictions, tuple)):
                    predictions = predictions[0]
            except:
                print('Fallo, segundo intento')
                try:
                    if len(Clasificadores[s][1]) > 0:
                        Clasificadores[s][0].fit(X_train, Y_train, **Clasificadores[s][1])
                    else:
                        Clasificadores[s][0].fit(X_train, Y_train)
                    predictions = Clasificadores[s][0].predict(X_test)
                    if (isinstance(predictions, tuple)):
                        predictions = predictions[0]
                    print('OK')
                except:
                    print('Fallo en calculo')
            for l,p in enumerate(test_index): 
                isCorrectLabel[s][p] = (Y_test.T[0][l] == np.sign(predictions[l]))

    if votacion == 'maxVotos':
        noisyBags = []
        for n in range(0,len(labels_f)):
            aux = 0
            for m in range(0,len(Clasificadores)):
                if not isCorrectLabel[m][n]:
                    aux = aux+1
            if aux > len(Clasificadores)/2:
                noisyBags.append(n)
    if votacion == 'consenso':
        noisyBags = []
        for n in range(0,len(labels_f)):
            aux = True
            for m in range(0,len(Clasificadores)):
                if aux:
                    if isCorrectLabel[m][n]:
                        aux = False
            if aux:
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
    print('\t\t\t=>Elementos eliminados : '+str(len(noisyBags)))
    X_train_NoNy = [bags_f[i] for i in nonNoisyBags]
    Y_train_NoNy = labels_f[nonNoisyBags]
    return X_train_NoNy,Y_train_NoNy

def filtrado_final(X_train,Y_train,X_test,Y_test):  
    Clasificadores = clasif()
    results = np.zeros((len(Clasificadores),2))
    for s,cl in enumerate(Clasificadores):
#        print('\t\t\t\t-->Clasificador :'+str(cl[2]))
        try:
            if len(Clasificadores[s][1]) > 0:
                Clasificadores[s][0].fit(X_train, Y_train, **Clasificadores[s][1])
            else:
                Clasificadores[s][0].fit(X_train, Y_train)
            predictions = Clasificadores[s][0].predict(X_test) 
            if (isinstance(predictions, tuple)):
                predictions = predictions[0]
            accuracie = (100 * np.average(Y_test.T == np.sign(predictions))) 
            auc_score = (100 * roc_auc_score_FIXED(Y_test,predictions))
        except:
            print('Fallo, segundo intento')
          
            try:
                if len(Clasificadores[s][1]) > 0:
                    Clasificadores[s][0].fit(X_train, Y_train, **Clasificadores[s][1])
                else:
                    Clasificadores[s][0].fit(X_train, Y_train)
                predictions = Clasificadores[s][0].predict(X_test) 
                if (isinstance(predictions, tuple)):
                    predictions = predictions[0]
                accuracie = (100 * np.average(Y_test.T == np.sign(predictions)))   
                auc_score = (100 * roc_auc_score_FIXED(Y_test,predictions))
                print('OK')     
            except:
                print('Fallo en calculo')      
        results[s][0] = accuracie
        results[s][1] = auc_score
#        print('\t\t\t\t\t Precisión: '+ str(accuracie)+'%\n\t\t\t\t\t Roc Score: '+ str(auc_score))
    return results

def cla_filter():
    aux = []
    resul1 = [[],[],[],[],[],[],[]]
    resul2 = [[],[],[],[],[],[],[]]
    resul3 = [[],[],[],[],[],[],[]]
    resul4 = [[],[],[],[],[],[],[]]
    resul5 = [[],[],[],[],[],[],[]]
    resul6 = [[],[],[],[],[],[],[]]
    resul7 = [[],[],[],[],[],[],[]]
    resul8 = [[],[],[],[],[],[],[]]
    resul9 = [[],[],[],[],[],[],[]]
    roc_m_1 = [[],[],[],[],[],[],[]]
    roc_m_2 = [[],[],[],[],[],[],[]]
    roc_m_3 = [[],[],[],[],[],[],[]]
    roc_m_4 = [[],[],[],[],[],[],[]]
    roc_m_5 = [[],[],[],[],[],[],[]]
    roc_m_6 = [[],[],[],[],[],[],[]]
    roc_m_7 = [[],[],[],[],[],[],[]]
    roc_m_8 = [[],[],[],[],[],[],[]]
    roc_m_9 = [[],[],[],[],[],[],[]]
    SMILaMax = [simpleMIL(),{'type': 'max'},'MIL max',resul1,roc_m_1]
    SMILaMin = [simpleMIL(),{'type': 'min'},'MIL min',resul2,roc_m_2]
    SMILaExt = [simpleMIL(),{'type': 'extreme'},'MIL Extreme',resul3,roc_m_3]
    BOW_clas = [BOW(),{'k':90,'covar_type':'diag','n_iter':20},'BOW',resul4,roc_m_4]
    CKNN_cla = [CKNN(),{'references': 3, 'citers': 5},'CKNN',resul5,roc_m_5]
    maxDD_cl = [maxDD(),{},'DIVERSE DENSITY',resul6,roc_m_6]
    EMDD_cla = [EMDD(),{},'EM-DD',resul7,roc_m_7]
    MILB_cla = [MILBoost(),{},'MILBOOST',resul8,roc_m_8]
    MILES_cl = [MILES(),{},'MILES',resul9,roc_m_9]
    aux.append(SMILaMax)
    aux.append(SMILaMin)
    aux.append(SMILaExt)
#    aux.append(BOW_clas)
#    aux.append(CKNN_cla)
#    aux.append(maxDD_cl)
#    aux.append(EMDD_cla)
#    aux.append(MILB_cla)
#    aux.append(MILES_cl)
    return aux