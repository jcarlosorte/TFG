pyMIL-BNF: Multiple Instance Learning Bag Noise Filters library in Python
=====================================================

Multiple-Instance Learning Python Proyect by Juan Carlos Orte (<jcarlosorte@correo.ugr.es>)

Abstract
--------

The rise of artificial vision techniques in our days is enhancing the techniques of Data Science related to the classification of objects. One of the most influential paradigms in the objects classification is the Multiple Instance Learning (MIL), where different representations are available for the same object.

However, the classification of MIL objects presents challenges today, since the identification of objects is not exact and erroneous representations (noisy) or even incorrect labels can be collected. In critical systems such as autonomous driving, biometric surveys and others, these errors can be catastrophic.

In this project, the development of a library of noise cleaning techniques for MIL written in Python is proposed. There are no proposals of this nature even in the specialized literature, so this project includes a research aspect.


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

![Interfaz pyMIL-BNF](https://i.ibb.co/dm5QFTN/app5.png)

by running use in menu file manually, ex:

 ```import sys,os
os.chdir('MILpy')
sys.path.append(os.path.realpath('..'))
from filters import IPF
folds = 5
votacion = 'consenso'
DataSet = ['tiger_scaled']
ruido = [0,5]
print('********** MIL-Iterative Partitioning Filter **********')
IPF.IPF(DataSet,votacion,folds,ruido)```


Disclaimer
--------
This Python proyect implementation is inspired by MILpy toolbox. 

Data from http://www.miproblems.org/datasets/

Cannot guarantee any support for this software.

License
--------
Copyright 2020 by UGR - Universidad de Granada
Permission to use, copy, or modify these programs and their documentation for educational and research purposes only and without fee is hereby granted, provided that this copyright notice appears on all copies and supporting documentation.  For any other uses of this software, in original or modified form, including but not limited to distribution in whole or in part, specific prior permission must be obtained from Universidad de Granada. These programs shall not be used, rewritten, or adapted as the basis of a commercial software or hardware product without first obtaining appropriate licenses from the Universidad de Granada.  Universidad de Granada makes no representations about the suitability of this software for any purpose.  It is provided "as is" without express or implied warranty.

Questions and Issues
--------------------

If you find any bugs or have any questions about this code, please create an
issue on [GitHub](https://github.com/jcarlosorte/pyMIL-BNF/issues), or contact Juan Carlos
Orte at <jcarlosorte@correo.ugr.es>. Of course, I cannot guarantee any support for
this software.

