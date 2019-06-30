import sys,os

os.chdir('MILpy')
sys.path.append(os.path.realpath('..'))

from filters import EF
from filters import CVCF
from filters import IPF

folds = 5
votacion = 'consenso'
DataSet = ['tiger_scaled']
ruido = [0]

#print('********** MIL-Ensemble Filter **********')
#EF.EF(DataSet,votacion,folds,ruido)
#print('********** MIL-CV Committees Filter **********')
#CVCF.CVcF(DataSet,votacion,folds,ruido)
print('********** MIL-Iterative Partitioning Filter **********')
IPF.IPF(DataSet,votacion,folds,ruido)