# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:27:50 2019

@author: Administrador
"""




import sys,os
import warnings
os.chdir('F:/CLASE/TFG/BibliotecasPython/MILpy')
sys.path.append(os.path.realpath('..'))
from sklearn.utils import shuffle
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
from MILpy.Algorithms.MILES import MILES
from MILpy.Algorithms.BOW import BOW


bags,labels,X = load_data('musk2_scaled')
folds = 5
runs = 1
print(labels)


cknn_classifier = CKNN() 
parameters_cknn = {'references': 3, 'citers': 5}
print '\n========= CKNN RESULT ========='
AUC = []
ACCURACIE=[]
for i in range(runs):
    print '\n run #'+ str(i)
    bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
    accuracie, results_accuracie, auc,results_auc, elapsed   = mil_cross_val(bags=bags,labels=labels.ravel(), model=cknn_classifier, folds=folds,parameters=parameters_cknn, timer=True)
    print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
    AUC.append(auc)
    ACCURACIE.append(accuracie)
print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))
