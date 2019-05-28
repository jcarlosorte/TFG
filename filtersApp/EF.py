# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:37:39 2019

@author: Juan Carlos
"""

#imports
import warnings,copy
from MILpy.data.load_data import load_data
import random as rand
import numpy as np
import pandas as pd
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
from MILpy.Algorithms.BOW import BOW

def EF(b,votacion,folds,ruido,clasif_O,clasif_F):
    for DataSet in b:
        bags,labels,X = load_data(DataSet)
        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, len(labels)-1))
        skf = StratifiedKFold(n_splits=folds)
#        dataAcc = np.zeros((len(b),len(ruido),folds,2))
        print('\n\tDATASET: '+str(DataSet)+'\n')
        for ny,k in enumerate(ruido):
            print('\t\t=>RUIDO : '+str(k))
            file_data = '../filtersApp/tabla.csv'
            data = {}
            data['EF'] = []
            data['Original'] = []
            data['Filtro_1'] = []
#            data['Filtro_2'] = []
            fold = 1
            results_Fil = []
#            results_Fil2 = []
            results_Ori = []
            Clasificadores_fake = clasif()
            Clasificadores = []
            for s,cl in enumerate(Clasificadores_fake):
                if str(cl[2]) == clasif_O:
                    Clasificadores.append(cl)
            for train_index, test_index in skf.split(bags, labels.reshape(len(labels))):
                print('\t\t\t=>FOLD : '+str(fold))
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
                num = int(clasif_F) + 1
                X_train_NoNy,Y_train_NoNy = mil_cv_filter_ef(X_train,Y_train,folds,votacion,num)
#                num = 2
#                X_train_NoNy2,Y_train_NoNy2 = mil_cv_filter_ef(X_train,Y_train,folds,votacion,num)
#                print('\t\t\t=>Original')
                results_Ori.append(filtrado_final(X_train,Y_train,X_test,Y_test,Clasificadores))
#                print('\t\t\t=>Filtrado')
                results_Fil.append(filtrado_final(X_train_NoNy,Y_train_NoNy,X_test,Y_test,Clasificadores))
#                results_Fil2.append(filtrado_final(X_train_NoNy2,Y_train_NoNy2,X_test,Y_test))
                fold = fold + 1
            Clasificadores_filtro = ""
            if clasif_F == "0":
                Clasificadores_filtro = cla_filter()
            elif clasif_F == "1":
                Clasificadores_filtro = cla_filter2()
            results_accuracie_F = []
            results_auc_F = []
#            results_accuracie_F2 = []
#            results_auc_F2 = []
            results_accuracie_O = []
            results_auc_O = []
            for h in range(0,len(Clasificadores)):
                print('\t\t\t\t-->Clasificador :'+str(Clasificadores[h][2]))
                data['EF'].append(str(Clasificadores[h][2]))
                for g in range(0,folds):
                    results_accuracie_F.append(results_Fil[g][h][0])
                    results_auc_F.append(results_Fil[g][h][1])
#                    results_accuracie_F2.append(results_Fil2[g][h][0])
#                    results_auc_F2.append(results_Fil2[g][h][1])
                    results_accuracie_O.append(results_Ori[g][h][0])
                    results_auc_O.append(results_Ori[g][h][1])
                print('\t\t\t\t\t-->Original')
                print('\t\t\t\t\t Precision: '+ str(np.mean(results_accuracie_O))+'%')
                data['Original'].append(np.mean(results_accuracie_O))
                print('\t\t\t\t\t Roc Score: '+ str(np.mean(results_auc_O)))
                print('\t\t\t\t\t-->Filtrado por ')
                for h,cl_f0 in enumerate(Clasificadores_filtro):
                    print('\t\t\t\t\t * '+str(cl_f0[2]))
#                print('\t\t\t\t\t-->Filtrado')
                print('\t\t\t\t\t Precision: '+ str(np.mean(results_accuracie_F))+'%')
                data['Filtro_1'].append(np.mean(results_accuracie_F))
                print('\t\t\t\t\t Roc Score: '+ str(np.mean(results_auc_F)))
                
#                for h,cl_f0 in enumerate(Clasificadores_filtro2):
#                    print('\t\t\t\t\t * '+str(cl_f0[2]))
#                print('\t\t\t\t\t-->Filtrado')
#                print('\t\t\t\t\t Precision: '+ str(np.mean(results_accuracie_F2))+'%')
#                data['Filtro_2'].append(np.mean(results_accuracie_F2))
#                print('\t\t\t\t\t Roc Score: '+ str(np.mean(results_auc_F2)))
            df = pd.DataFrame(data)
            df.to_csv(file_data, sep=';')

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
    roc_m_1 = [[],[],[],[],[],[],[]]
    roc_m_2 = [[],[],[],[],[],[],[]]
    roc_m_3 = [[],[],[],[],[],[],[]]
    roc_m_4 = [[],[],[],[],[],[],[]]
    roc_m_5 = [[],[],[],[],[],[],[]]
    roc_m_6 = [[],[],[],[],[],[],[]]
    roc_m_7 = [[],[],[],[],[],[],[]]
    roc_m_8 = [[],[],[],[],[],[],[]]
    SMILaMax = [simpleMIL(),{'type': 'max'},'MIL max',resul1,roc_m_1]
    SMILaMin = [simpleMIL(),{'type': 'min'},'MIL min',resul2,roc_m_2]
    SMILaExt = [simpleMIL(),{'type': 'extreme'},'MIL Extreme',resul3,roc_m_3]
    BOW_clas = [BOW(),{'k':90,'covar_type':'diag','n_iter':20},'BOW',resul4,roc_m_4]
    CKNN_cla = [CKNN(),{'references': 3, 'citers': 5},'CKNN',resul5,roc_m_5]
    maxDD_cl = [maxDD(),{},'DIVERSE DENSITY',resul6,roc_m_6]
    EMDD_cla = [EMDD(),{},'EM-DD',resul7,roc_m_7]
    MILB_cla = [MILBoost(),{},'MILBOOST',resul8,roc_m_8]    
    aux.append(SMILaMax)
    aux.append(SMILaMin)
    aux.append(SMILaExt)
    aux.append(BOW_clas)
    aux.append(CKNN_cla)   
    aux.append(maxDD_cl)   
    aux.append(EMDD_cla)   
    aux.append(MILB_cla)
    return aux

def mil_cv_filter_ef(bags_f,labels_f,folds,votacion,num):
#    print('\t\t\tFiltrando...')
    if num == 1:
        Clasificadores = cla_filter()
    else:
        Clasificadores = cla_filter2()
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
                    Clasificadores[s][0].fit(bags_f, labels_f)
                predictions = Clasificadores[s][0].predict(X_test)
                if (isinstance(predictions, tuple)):
                    predictions = predictions[0]
            except:
                print('Fallo, segundo intento')
                try:
                    if len(Clasificadores[s][1]) > 0:
                        Clasificadores[s][0].fit(X_train, Y_train, **Clasificadores[s][1])
                    else:
                        Clasificadores[s][0].fit(bags_f, labels_f)
                    predictions = Clasificadores[s][0].predict(X_test)
                    if (isinstance(predictions, tuple)):
                        predictions = predictions[0]
                    print('OK')
                except:
                    print('Posible fallo en bolsa...')
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
                        print('Fallo')
            for l,p in enumerate(test_index):
                try:
                    isCorrectLabel[s][p] = (Y_test.T[0][l] == np.sign(predictions[l]))
                except IndexError:
                    print("Fallo en ultimo indice!")
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
    print('\t\t\t=>Elementos eliminados con Filter '+str(num + 1)+': '+str(len(noisyBags)))
    X_train_NoNy = [bags_f[i] for i in nonNoisyBags]
    Y_train_NoNy = labels_f[nonNoisyBags]
    return X_train_NoNy,Y_train_NoNy

def filtrado_final(X_train,Y_train,X_test,Y_test,cl_fil):  
    Clasificadores = cl_fil
    results = np.zeros((len(Clasificadores),2))
    aux_lab = True
    if len(np.unique(Y_train)) == 1:
        if Y_train[0] == 0:
            for b in range(0,len(Y_test)):
                if aux_lab:
                    if Y_test[b] == 1:
                        aux_Y_train = copy.copy(Y_test[b])
                        aux_X_train = copy.copy(X_test[b])
                        Y_test[b] = copy.copy(Y_train[0])
                        X_test[b] = copy.copy(X_train[0])
                        Y_train[0] = copy.copy(aux_Y_train)
                        X_train[0] = copy.copy(aux_X_train)
                        aux_lab = False
        else:
            for b in range(0,len(Y_test)):
                if aux_lab:
                    if Y_test[b] == 0:
                        aux_Y_train = copy.copy(Y_test[b])
                        aux_X_train = copy.copy(X_test[b])
                        Y_test[b] = copy.copy(Y_train[0])
                        X_test[b] = copy.copy(X_train[0])
                        Y_train[0] = copy.copy(aux_Y_train)
                        X_train[0] = copy.copy(aux_X_train)
                        aux_lab = False
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
    roc_m_1 = [[],[],[],[],[],[],[]]
    roc_m_2 = [[],[],[],[],[],[],[]]
    roc_m_3 = [[],[],[],[],[],[],[]]
    roc_m_4 = [[],[],[],[],[],[],[]]
    roc_m_5 = [[],[],[],[],[],[],[]]
    roc_m_6 = [[],[],[],[],[],[],[]]
    roc_m_7 = [[],[],[],[],[],[],[]]
    roc_m_8 = [[],[],[],[],[],[],[]]
    SMILaMax = [simpleMIL(),{'type': 'max'},'MIL max',resul1,roc_m_1]
    SMILaMin = [simpleMIL(),{'type': 'min'},'MIL min',resul2,roc_m_2]
    SMILaExt = [simpleMIL(),{'type': 'extreme'},'MIL Extreme',resul3,roc_m_3]
    BOW_clas = [BOW(),{'k':90,'covar_type':'diag','n_iter':20},'BOW',resul4,roc_m_4]
    CKNN_cla = [CKNN(),{'references': 3, 'citers': 5},'CKNN',resul5,roc_m_5]
    maxDD_cl = [maxDD(),{},'DIVERSE DENSITY',resul6,roc_m_6]
    EMDD_cla = [EMDD(),{},'EM-DD',resul7,roc_m_7]
    MILB_cla = [MILBoost(),{},'MILBOOST',resul8,roc_m_8]
#    aux.append(SMILaMax)
#    aux.append(SMILaMin)
#    aux.append(SMILaExt)
    aux.append(BOW_clas)
    aux.append(CKNN_cla)
#    aux.append(maxDD_cl)
    aux.append(EMDD_cla)
#    aux.append(MILB_cla)
    return aux

def cla_filter2():
    aux = []
    resul1 = [[],[],[],[],[],[],[]]
    resul2 = [[],[],[],[],[],[],[]]
    resul3 = [[],[],[],[],[],[],[]]
    resul4 = [[],[],[],[],[],[],[]]
    resul5 = [[],[],[],[],[],[],[]]
    resul6 = [[],[],[],[],[],[],[]]
    resul7 = [[],[],[],[],[],[],[]]
    resul8 = [[],[],[],[],[],[],[]]
    roc_m_1 = [[],[],[],[],[],[],[]]
    roc_m_2 = [[],[],[],[],[],[],[]]
    roc_m_3 = [[],[],[],[],[],[],[]]
    roc_m_4 = [[],[],[],[],[],[],[]]
    roc_m_5 = [[],[],[],[],[],[],[]]
    roc_m_6 = [[],[],[],[],[],[],[]]
    roc_m_7 = [[],[],[],[],[],[],[]]
    roc_m_8 = [[],[],[],[],[],[],[]]
    SMILaMax = [simpleMIL(),{'type': 'max'},'MIL max',resul1,roc_m_1]
    SMILaMin = [simpleMIL(),{'type': 'min'},'MIL min',resul2,roc_m_2]
    SMILaExt = [simpleMIL(),{'type': 'extreme'},'MIL Extreme',resul3,roc_m_3]
    BOW_clas = [BOW(),{'k':90,'covar_type':'diag','n_iter':20},'BOW',resul4,roc_m_4]
    CKNN_cla = [CKNN(),{'references': 3, 'citers': 5},'CKNN',resul5,roc_m_5]
    maxDD_cl = [maxDD(),{},'DIVERSE DENSITY',resul6,roc_m_6]
    EMDD_cla = [EMDD(),{},'EM-DD',resul7,roc_m_7]
    MILB_cla = [MILBoost(),{},'MILBOOST',resul8,roc_m_8]
    aux.append(SMILaMax)
#    aux.append(SMILaMin)
    aux.append(SMILaExt)
#    aux.append(BOW_clas)
#    aux.append(CKNN_cla)
    aux.append(maxDD_cl)
#    aux.append(EMDD_cla)
#    aux.append(MILB_cla)
    return aux