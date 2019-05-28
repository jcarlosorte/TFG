# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:11:00 2019

@author: Usuario
"""

import sys,os,warnings

os.chdir('../../../../MILpy')
sys.path.append(os.path.realpath('..'))

warnings.filterwarnings('ignore')
#from funciones import fvc
from filters import EF
from filters import CVCF
from filters import IPF
folds = 5
votacion = 'consenso'
DataSet = ['musk1_scaled','musk2_scaled','elephant_scaled','fox_scaled','mutagenesis1_scaled','mutagenesis2_scaled','tiger_scaled']
#ruido = [0,5,10,15,20,25,30]
ruido = [20]
#print('********** Crear dataset con ruido **********')
#fvc.fvc_part(DataSet,folds,ruido)
print('********** Ensemble Filter por '+str(votacion)+'**********')
EF.EF(DataSet,votacion,folds,ruido)
#print('********** CV Committees Filter por '+str(votacion)+'**********')
#CVCF.CVcF(DataSet,votacion,folds,ruido)
#print('********** Iterative Partitioning Filter por '+str(votacion)+'**********')
#IPF.IPF(DataSet,votacion,folds,ruido)
#votacion = 'maxVotos'
#print('********** Ensemble Filter por '+str(votacion)+'**********')
#EF.EF(DataSet,votacion,folds,ruido)
#print('********** CV Committees Filter por '+str(votacion)+'**********')
#CVCF.CVcF(DataSet,votacion,folds,ruido)
#print('********** Iterative Partitioning Filter por '+str(votacion)+'**********')
#IPF.IPF(DataSet,votacion,folds,ruido)