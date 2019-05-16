# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:11:00 2019

@author: Usuario
"""

import sys,os,warnings

os.chdir('C:/Users/Administrador/Documents/GitHub/TFG/MILpy')
sys.path.append(os.path.realpath('..'))

warnings.filterwarnings('ignore')
from funciones import fvc
from filters import EF
from filters import CVCF
from filters import IPF
folds = 5
votacion = 'consenso'
DataSet = ['Tiger_scaled']
#ruido = [0,5,10,15,20,25,30]
ruido = [0]
#print('********** Crear dataset con ruido **********')
#fvc.fvc_part(DataSet,folds,ruido)
#print('********** Ensemble Filter **********')
EF.EF(DataSet,votacion,folds,ruido)
#print('********** CV Committees Filter **********')
#CVCF.CVcF(DataSet,votacion,folds,ruido)
#print('********** Iterative Partitioning Filter **********')
#IPF.IPF(DataSet,votacion,folds,ruido)