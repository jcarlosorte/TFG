

import sys,os
import warnings
os.chdir('C:/Users/Administrador/Documents/GitHub/TFG/MILpy')
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
#from MILpy.Algorithms.MILES import MILES
from MILpy.Algorithms.BOW import BOW

folds = 5
runs = 1
#DataSet = ['musk1_scaled','Musk2_scaled','Elephant_scaled','Fox_scaled','mutagenesis1_scaled','mutagenesis2_scaled','Tiger_scaled']
#DataSet = ['birds_WIWR_scaled','birds_BRCR_scaled']
DataSet = ['Elephant_scaled']
for j in DataSet:
    print '\n========= DATASET: ',j,' ========='
    bags,labels,X = load_data(j)
#    SMILa = simpleMIL()
#    parameters_smil = {'type': 'max'}
#    print '\n========= SIMPLE MIL RESULT [MAX] ========='
#    AUC = []
#    ACCURACIE=[]
#    for i in range(runs):
##        print '\n run #'+ str(i)
#        #Shuffle Data
#        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
#        accuracie, results_accuracie, auc,results_auc, elapsed  = mil_cross_val(bags=bags,labels=labels.ravel(), model=SMILa, folds=folds, parameters=parameters_smil, timer = True)
##        print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
#        AUC.append(auc)
#        ACCURACIE.append(accuracie)
#    print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))
#    parameters_smil = {'type': 'min'}
#    print '\n========= SIMPLE MIL RESULT [MIN] ========='
#    AUC = []
#    ACCURACIE=[]
#    for i in range(runs):
##        print '\n run #'+ str(i)
#        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
#        accuracie, results_accuracie, auc,results_auc, elapsed   = mil_cross_val(bags=bags,labels=labels.ravel(), model=SMILa, folds=folds,parameters=parameters_smil, timer=True)
##        print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
#        AUC.append(auc)
#        ACCURACIE.append(accuracie)
#    print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))
#    parameters_smil = {'type': 'extreme'}
#    print '\n========= SIMPLE MIL RESULT [MIN] ========='
#    AUC = []
#    ACCURACIE=[]
#    for i in range(runs):
##        print '\n run #'+ str(i)
#        #Shuffle Data
#        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
#        accuracie, results_accuracie, auc,results_auc, elapsed = mil_cross_val(bags=bags,labels=labels.ravel(), model=SMILa, folds=folds,parameters=parameters_smil, timer=True)
##        print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
#        AUC.append(auc)
#        ACCURACIE.append(accuracie)
#    print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE)) 
#    bow_classifier = BOW() 
#    parameters_bow = {'k':100,'covar_type':'diag','n_iter':20}
#    print '\n========= BAG OF WORDS RESULT ========='
#    AUC = []
#    ACCURACIE=[]
#    for i in range(runs):
##        print '\n run #'+ str(i)
#        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
#        accuracie, results_accuracie, auc,results_auc, elapsed = mil_cross_val(bags=bags,labels=labels.ravel(), model=bow_classifier, folds=folds,parameters=parameters_bow, timer=True)
##        print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
#        AUC.append(auc)
#        ACCURACIE.append(accuracie)
#    print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))
#    cknn_classifier = CKNN() 
#    parameters_cknn = {'references': 3, 'citers': 5}
#    print '\n========= CKNN RESULT ========='
#    AUC = []
#    ACCURACIE=[]
#    for i in range(runs):
##        print '\n run #'+ str(i)
#        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
#        accuracie, results_accuracie, auc,results_auc, elapsed   = mil_cross_val(bags=bags,labels=labels.ravel(), model=cknn_classifier, folds=folds,parameters=parameters_cknn, timer=True)
##        print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
#        AUC.append(auc)
#        ACCURACIE.append(accuracie)
#    print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))
#    maxDD_classifier = maxDD()
#    print '\n========= DIVERSE DENSITY RESULT========='
#    AUC = []
#    ACCURACIE=[]
#    for i in range(runs):
##        print '\n run #'+ str(i)
#        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
#        accuracie, results_accuracie, auc,results_auc, elapsed = mil_cross_val(bags=bags,labels=labels.ravel(), model=maxDD_classifier, folds=folds,parameters={}, timer=True)
##        print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
#        AUC.append(auc)
#        ACCURACIE.append(accuracie)
#    print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))
    emdd_classifier = EMDD()
    print '\n========= EM-DD RESULT ========='
    AUC = []
    ACCURACIE=[]
    for i in range(runs):
#        print '\n run #'+ str(i)
        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
        accuracie, results_accuracie, auc,results_auc, elapsed = mil_cross_val(bags=bags,labels=labels.ravel(), model=emdd_classifier, folds=folds,parameters={}, timer=True)
#        print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
        AUC.append(auc)
        ACCURACIE.append(accuracie)
    print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))
    milboost_classifier = MILBoost()
    print '\n========= MILBOOST RESULT ========='
    AUC = []
    ACCURACIE=[]
    for i in range(runs):
#        print '\n run #'+ str(i)
        bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
        accuracie, results_accuracie, auc,results_auc, elapsed = mil_cross_val(bags=bags,labels=labels, model=milboost_classifier, folds=folds,parameters={}, timer=True)
#        print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
        AUC.append(auc)
        ACCURACIE.append(accuracie)
    print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))


print '\n========= FIN ========='   
                         #######
#                            #MILES#
#                            #######
#bags,labels,_ = load_data('data_gauss')  #Gaussian data
#miles_classifier = MILES()
#
#print '\n========= MILES RESULT ========='
#AUC = []
#ACCURACIE=[]
#for i in range(runs):
#    print '\n run #'+ str(i)
#    bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))   
#    accuracie, results_accuracie, auc,results_auc, elapsed = mil_cross_val(bags=bags,labels=labels, model=miles_classifier, folds=folds,parameters={}, timer=True)
#    print '\n'+'AUC: ' + str(auc)+'\n'+'Accuracie: '+ str(accuracie)+'\n'+'Elapsed: '+ str(round(elapsed,2))
#    AUC.append(auc)
#    ACCURACIE.append(accuracie)
#print '\n MEAN AUC: '+ str(np.mean(AUC)) + '\n MEAN ACCURACIE: '+ str(np.mean(ACCURACIE))
#    





