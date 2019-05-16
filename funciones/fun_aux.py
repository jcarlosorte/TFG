# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:27:08 2019

@author: Juan Carlos Orte
"""
import csv,warnings
import numpy as np
import random as rand
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
#Import Algorithms 
from MILpy.Algorithms.simpleMIL import simpleMIL
from MILpy.Algorithms.MILBoost import MILBoost
from MILpy.Algorithms.maxDD import maxDD
from MILpy.Algorithms.CKNN import CKNN
from MILpy.Algorithms.EMDD import EMDD
from MILpy.Algorithms.MILES import MILES
from MILpy.Algorithms.BOW import BOW


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
#    aux.append(SMILaExt)
    aux.append(BOW_clas)
#    aux.append(CKNN_cla)
#    aux.append(maxDD_cl)
#    aux.append(EMDD_cla)
#    aux.append(MILB_cla)
#    aux.append(MILES_cl)
    return aux
def loadNoisy(dataset,noisy_file,fold_file):
    carpeta = '../dataNoisy/'
    filename1 = 'X_train_bags.csv'
    filename2 = 'Y_train_labels.csv'
    filename3 = 'X_test_bags.csv'
    filename4 = 'Y_test_labels.csv'
  
#    print('==================reader')
    if noisy_file > 0:
        try:
    #        print('==================X_train_bags')
            with open(carpeta+dataset+'/fold_'+str(fold_file)+'/Noisy_'+str(noisy_file)+'/'+filename1) as File1:
                reader = csv.reader(File1, delimiter=',')
                data = list(reader)
                data = np.array(data).astype(int)
                data = data.reshape((1, len(data)))
                train_index = data[0]
    #            print(data[0])
    #        print('==================Y_train_labels')
            with open(carpeta+dataset+'/fold_'+str(fold_file)+'/Noisy_'+str(noisy_file)+'/'+filename2) as File2:
                reader = csv.reader(File2, delimiter=',')
                data = list(reader)
                Y_train = np.array(data).astype(int)
    #            data = data.reshape((1, len(data)))
    #            print(data)
    #        print('==================X_test_bags')
            with open(carpeta+dataset+'/fold_'+str(fold_file)+'/Noisy_'+str(noisy_file)+'/'+filename3) as File3:
                reader = csv.reader(File3, delimiter=',')
                data = list(reader)
                data = np.array(data).astype(int)
                data = data.reshape((1, len(data)))
                test_index = data[0]
    #            print(data[0])
    #        print('==================Y_test_labels')
            with open(carpeta+dataset+'/fold_'+str(fold_file)+'/Noisy_'+str(noisy_file)+'/'+filename4) as File4:
                reader = csv.reader(File4, delimiter=',')
                data = list(reader)
                Y_test = np.array(data).astype(int)
    #            data = data.reshape((1, len(data)))
    #            print(data)
        except:
            print('No existe el archivo indicado')
    else:
        try:
    #        print('==================X_train_bags')
            with open(carpeta+dataset+'/fold_'+str(fold_file)+'/Original/'+filename1) as File1:
                reader = csv.reader(File1, delimiter=',')
                data = list(reader)
                data = np.array(data).astype(int)
                data = data.reshape((1, len(data)))
                train_index = data[0]
    #            print(data[0])
    #        print('==================Y_train_labels')
            with open(carpeta+dataset+'/fold_'+str(fold_file)+'/Original/'+filename2) as File2:
                reader = csv.reader(File2, delimiter=',')
                data = list(reader)
                Y_train = np.array(data).astype(int)
    #            data = data.reshape((1, len(data)))
    #            print(data)
    #        print('==================X_test_bags')
            with open(carpeta+dataset+'/fold_'+str(fold_file)+'/Original/'+filename3) as File3:
                reader = csv.reader(File3, delimiter=',')
                data = list(reader)
                data = np.array(data).astype(int)
                data = data.reshape((1, len(data)))
                test_index = data[0]
    #            print(data[0])
    #        print('==================Y_test_labels')
            with open(carpeta+dataset+'/fold_'+str(fold_file)+'/Original/'+filename4) as File4:
                reader = csv.reader(File4, delimiter=',')
                data = list(reader)
                Y_test = np.array(data).astype(int)
    #            data = data.reshape((1, len(data)))
    #            print(data)
        except:
            print('No existe el archivo indicado')
    return train_index,Y_train,test_index,Y_test
def mil_cross_noisy(nonNoisyBags,DataSet,bags,labels,folds):  
    Clasificadores = clasif()
    
    bags_noNoisy = [bags[d] for d in nonNoisyBags] 
    labels_noNoisy = labels[nonNoisyBags]

    bags_noNoisy,labels_noNoisy = shuffle(bags_noNoisy, labels_noNoisy, random_state=rand.randint(0, 100))
    skf = StratifiedKFold(n_splits=folds)
    
    
    for s,cl in enumerate(Clasificadores):
        fold = 1
        print('\t\t\t-->Clasificador :'+str(cl[2]))
        results_accuracie = []
        results_auc = []
        for train_index, test_index in skf.split(bags_noNoisy, labels_noNoisy.reshape(len(labels_noNoisy))):
#            print('========= Fold :'+str(fold)+' =========') 
            X_train = [bags_noNoisy[i] for i in train_index]        
            Y_train = labels_noNoisy[train_index]
            X_test  = [bags_noNoisy[i] for i in test_index]
            Y_test  = labels_noNoisy[test_index]
            try:
                if len(Clasificadores[s][1]) > 0:
                    Clasificadores[s][0].fit(X_train, Y_train, **Clasificadores[s][1])
                else:
                    Clasificadores[s][0].fit(X_train, Y_train)
                predictions = Clasificadores[s][0].predict(X_test) 
                if (isinstance(predictions, tuple)):
                    predictions = predictions[0]
                accuracie = np.average(Y_test.T == np.sign(predictions)) 
                results_accuracie.append(100 * accuracie)
                auc_score = roc_auc_score_FIXED(Y_test,predictions)  
                results_auc.append(100 * auc_score)
                
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
                    accuracie = np.average(Y_test.T == np.sign(predictions)) 
                    results_accuracie.append(100 * accuracie)
                    auc_score = roc_auc_score_FIXED(Y_test,predictions)  
                    results_auc.append(100 * auc_score)
                   
                    print('OK')
                   
                except:
                    print('Fallo en calculo')
            fold = fold+1
             
        print('\t\t\t\t Precisión Media: '+ str(np.mean(results_accuracie))+'%\n\t\t\t\t Media Roc Score: '+ str(np.mean(results_auc)))
            
def crearDataSet(noisyBags,bags,labels,DataSet,folds):
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
    mil_cross_noisy(nonNoisyBags,DataSet,bags,labels,folds)    
    
    
def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)    
    
def mil_cv_filter(bags_f,labels_f,folds,votacion):
#    print('\t\t\tFiltrando...')
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
    X_train_NoNy = [bags_f[i] for i in nonNoisyBags]
    Y_train_NoNy = labels_f[nonNoisyBags]
    return X_train_NoNy,Y_train_NoNy
        
def filtrado_final(X_train,Y_train,X_test,Y_test):  
    Clasificadores = clasif()
    for s,cl in enumerate(Clasificadores):
        print('\t\t\t\t-->Clasificador :'+str(cl[2]))
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
        print('\t\t\t\t\t Precisión: '+ str(accuracie)+'%\n\t\t\t\t\t Roc Score: '+ str(auc_score))