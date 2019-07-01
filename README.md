pyMIL-BNF: Multiple Instance Learning Bag Noise Filters library in Python
=====================================================

TFG Multiple-Instance Learning Python Proyect by Juan Carlos Orte (<jcarlosorte@ugr.es>)


Overview
--------
PyMIL-BNF contains a Python implementation of multiple-instance learning (MIL) framework and
uses the toolbox MILpy, the toolbox contains algorithms to train and evaluate MIL classifiers.

Stage of Development:
---------------------

->Development still on process

Contents
--------
The pyMIL-BNF package currently implements the following filters algorithms:

MIL-EF

MIL-CVCF

MIL-IPF

How to Use
----------
This package can be used in two ways:

by running the app

`py -3 filterApp.py`

![Interfaz pyMIL-BNF](https://github.com/jcarlosorte/TFG/blob/master/filtersApp/app5.png "pyMIL-BNF menu")

by running use in menu file manually, ex:

`import sys,os

os.chdir('MILpy')

sys.path.append(os.path.realpath('..'))

from filters import IPF

folds = 5

votacion = 'consenso'

DataSet = ['tiger_scaled']

ruido = [0,5]

print('********** MIL-Iterative Partitioning Filter **********')

IPF.IPF(DataSet,votacion,folds,ruido)`


Disclaimer
--------
This Python proyect implementation is inspired by MILpy toolbox. 

Data from http://www.miproblems.org/datasets/

Cannot guarantee any support for this software.

License
--------
Copyright 2018-2019 by UGR - Universidad de Granada (Ceuta)
Permission to use, copy, or modify these programs and their documentation for educational and research purposes only and without fee is hereby granted, provided that this copyright notice appears on all copies and supporting documentation.  For any other uses of this software, in original or modified form, including but not limited to distribution in whole or in part, specific prior permission must be obtained from Universidad de Granada. These programs shall not be used, rewritten, or adapted as the basis of a commercial software or hardware product without first obtaining appropriate licenses from the Universidad de Granada.  Universidad de Granada makes no representations about the suitability of this software for any purpose.  It is provided "as is" without express or implied warranty.

Questions and Issues
--------------------

If you find any bugs or have any questions about this code, please create an
issue on [GitHub](https://github.com/jcarlosorte/TFG/issues), or contact Gary
Doran at <jcarlosorte@correo.ugr.es>. Of course, I cannot guarantee any support for
this software.

