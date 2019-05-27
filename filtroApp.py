# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:11:00 2019

@author: Juan Carlos Orte Cardona
"""

import sys,os,warnings
from os import walk
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, \
QPushButton, QComboBox, QStyleFactory, QTableWidget, \
QTableWidgetItem, QGroupBox, QRadioButton, \
QSlider, QLCDNumber, QMessageBox
warnings.filterwarnings('ignore')
os.chdir('C:/Users/Administrador/Documents/GitHub/TFG/MILpy')
sys.path.append(os.path.realpath('..'))
import pandas as pd
from filtersApp import EF
from filtersApp import CVCF
from filtersApp import IPF

class CargaAlgoritmo(QThread):
    def __init__(self,ck_clasif, DataSet,votacion,folds,ruido,clasif,clasif_F):
        super().__init__()
        self._ck_clasif = ck_clasif
        self._DataSet = DataSet
        self._votacion = votacion
        self._folds = folds
        self._ruido = ruido
        self._clasif = clasif
        self._clasif_F = clasif_F
    def run(self):
        # Abrir la dirección de URL.
        if self._ck_clasif == 1:
            print("Ensemble Filter")
            EF.EF(self._DataSet,self._votacion,self._folds,self._ruido,self._clasif,self._clasif_F)
        elif self._ck_clasif == 2:
            print("Cross-Validation Committees Filter")
            CVCF.CVcF(self._DataSet,self._votacion,self._folds,self._ruido,self._clasif,self._clasif_F)
        elif self._ck_clasif == 3:
            print("Iterative Partitioning Filter")
            IPF.IPF(self._DataSet,self._votacion,self._folds,self._ruido,self._clasif,self._clasif_F)


class FilterTFG(QWidget):
    def __init__(self, parent=None):
            super(FilterTFG, self).__init__(parent)
            self.initUI()
            QApplication.setStyle(QStyleFactory.create('Fusion'))
    def initUI(self):
        self.addDataSet()
        self.addRadioBotonFilter()
        self.addSliderRuido()
        self.addTable()
        self.addEnviar()
        self.resize(430, 450)
        self.setWindowTitle('TFG Filtros MIL')

    def addTable(self):
        self.gbx_t = QGroupBox('Precisión:',self)
        self.gbx_t.setGeometry(190, 10, 230, 80)
        self.layout_t = QVBoxLayout(self)
        self.layout_t.setContentsMargins(0,0,0,0)
        # Create table
        self.tableWidget = QTableWidget()
        self.layout_t.addWidget(self.tableWidget)
        self.gbx_t.setLayout(self.layout_t)
        self.gbx_t.hide()
        
    def addDataSet(self):
        items = self.ls(os.getcwd()+str("/data/"))
        self.Ldt = QComboBox(self)
        self.Ldt.addItems(items)
        self.Ldt.move(10, 35)
        lbl = QLabel('Seleccionar DataSet:', self)
        lbl.move(10, 10)

    def addEnviar(self):
        self.label = QLabel("Cuando inicie el calculo es probable que tarde.", self)
        self.label.setGeometry(150, 420, 220, 25)
        self.btn1 = QPushButton("Enviar", self)
        self.btn1.move(40, 425)
        self.btn1.hide()
        self.btn1.pressed.connect(self.buttonClicked)
    def buttonClicked(self):
        self.file_data = '../filtersApp/tabla.csv'
        folds = 0
        votacion = ''
        DataSet = []
        ruido = []
        clasif = ""
        clasif_F = ""
        self.label.setText("Esto puede tardar.Calculando...")
        self.btn1.setEnabled(False)
        btn_txt = "\nDataSet : " + self.Ldt.currentText()
        DataSet.append(self.Ldt.currentText())
        ck_clasif = 0
        #Valor Filter
        for i, radio in enumerate(self.RbFilter):
            if radio.isChecked():
                btn_txt = btn_txt + "\nFiltro : " + radio.text()
                if radio.text() == "Ensemble Filter":
                    ck_clasif = 1
                    for j, radio2 in enumerate(self.RbClasif_ef):
                        if radio2.isChecked():
                            btn_txt = btn_txt + "\nClasificador : " + radio2.text()
                            clasif = radio2.text()
                    for p, radio4 in enumerate(self.RbClasif_F_ef):
                        if radio4.isChecked():
                            btn_txt = btn_txt + "\nClasificador Filtro : " + radio4.text()
                            clasif_F = str(p)
                elif radio.text() == "Cross-Validation Committees Filter":
                    ck_clasif = 2
                    for j, radio2 in enumerate(self.RbClasif_cvcf):
                        if radio2.isChecked():
                            btn_txt = btn_txt + "\nClasificador : " + radio2.text()
                            clasif = radio2.text()
                    for p, radio4 in enumerate(self.RbClasif_F_cvcf):
                        if radio4.isChecked():
                            btn_txt = btn_txt + "\nClasificador Filtro : " + radio4.text()
                            clasif_F = radio4.text()
                elif radio.text() == "Iterative Partitioning Filter":
                    ck_clasif = 3
                    for j, radio2 in enumerate(self.RbClasif_ipf):
                        if radio2.isChecked():
                            btn_txt = btn_txt + "\nClasificador : " + radio2.text()
                            clasif = radio2.text()
                    for p, radio4 in enumerate(self.RbClasif_F_ipf):
                        if radio4.isChecked():
                            btn_txt = btn_txt + "\nClasificador Filtro : " + radio4.text()
                            clasif_F = radio4.text()
        btn_txt = btn_txt + "\nRuido : " + str(int(self.num.value()))    
        ruido.append(int(self.num.value()))
        btn_txt = btn_txt + "\nFolds : " + str(int(self.num2.value()))
        folds = int(self.num2.value())
        for u, radio3 in enumerate(self.RbVotacion):
            if radio3.isChecked():
                btn_txt = btn_txt + "\nVotacion : " + radio3.text()
                if radio3.text() == "Consenso":
                    votacion = 'consenso'
                elif radio3.text() == "Mayoria":
                    votacion = 'maxVotos'
        
        QMessageBox.information(self, 'CONFIGURACIÓN', btn_txt) 
        
        self.cargaAlgoritmo = CargaAlgoritmo(ck_clasif,DataSet,votacion,folds,ruido,clasif,clasif_F)
        self.cargaAlgoritmo.finished.connect(self.cargaCompleta)
        self.cargaAlgoritmo.start()
        
    def cargaCompleta(self):
        QMessageBox.information(self, 'AVISO', "¡Completado!")
        self.label.setText("¡Completado!")
        self.loadCsv(self.file_data)
        self.btn1.setEnabled(True)
        del self.cargaAlgoritmo
        
    def loadCsv(self, fileName):
        df = pd.read_csv(filepath_or_buffer=fileName, sep=';', index_col=0)
#        df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df1 = df.columns
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels([df1[0], df1[1], df1[2]])
        for i, row in enumerate(df.values):
            rowPosition = self.tableWidget.rowCount()
            self.tableWidget.insertRow(rowPosition)
            for h, li in enumerate(row):
                self.tableWidget.setItem(i, h, QTableWidgetItem(str(li)))
        self.tableWidget.resizeColumnsToContents()
        self.gbx_t.show()
        
    def addRadioBotonFilter(self):
        gbx = QGroupBox('Filtros:', self)
        gbx.setGeometry(10, 95, 235, 100)
        # crear tres QRadioButton
        self.RbFilter = []
        self.radio1 = QRadioButton("Ensemble Filter")
        self.radio2 = QRadioButton("Cross-Validation Committees Filter")
        self.radio3 = QRadioButton("Iterative Partitioning Filter")
        self.radio1.toggled.connect(lambda:self.btnstate(self.radio1))
        self.radio2.toggled.connect(lambda:self.btnstate(self.radio2))
        self.radio3.toggled.connect(lambda:self.btnstate(self.radio3))
        self.RbFilter.append(self.radio1)
        self.RbFilter.append(self.radio2)
        self.RbFilter.append(self.radio3)
        # agregar los widgets al layout vertical
        self.vbox = QVBoxLayout(self)
        self.vbox.addWidget(self.radio1)
        self.vbox.addWidget(self.radio2)
        self.vbox.addWidget(self.radio3)
        
        gbx.setLayout(self.vbox)
        #EF----------------------------
        self.gbx2 = QGroupBox('Configuración de Filtros I:', self)
        self.gbx2.setGeometry(10, 200, 130, 220)
        self.vbox2 = QVBoxLayout(self)
        self.vbox2.addWidget(QLabel('Clasificador:', self))
        clasif_ef = EF.clasif()
        check1 = True
        self.RbClasif_ef = []
        for s,cl in enumerate(clasif_ef):
            if check1:
                butt = QRadioButton(str(cl[2]))
                butt.setChecked(True)
                self.vbox2.addWidget(butt)
                self.RbClasif_ef.append(butt)
                check1 = False
            else:
                butt = QRadioButton(str(cl[2]))
                self.vbox2.addWidget(butt)
                self.RbClasif_ef.append(butt)
        self.gbx2.setLayout(self.vbox2)
        self.gbx2.hide()        
        #CVCF----------------------------
        self.gbx3 = QGroupBox('Configuración de Filtros I:', self)
        self.gbx3.setGeometry(10, 200, 130, 220)
        self.vbox3 = QVBoxLayout(self)
        self.vbox3.addWidget(QLabel('Clasificador:', self))
        clasif_cvcf = CVCF.clasif()
        check2 = True
        self.RbClasif_cvcf = []
        for s,cl in enumerate(clasif_cvcf):
            if check2:
                butt = QRadioButton(str(cl[2]))
                butt.setChecked(True)
                self.vbox3.addWidget(butt)
                self.RbClasif_cvcf.append(butt)
                check2 = False
            else:
                butt = QRadioButton(str(cl[2]))
                self.vbox3.addWidget(butt)
                self.RbClasif_cvcf.append(butt)
        self.gbx3.setLayout(self.vbox3)
        self.gbx3.hide()
        #IPF----------------------------
        self.gbx4 = QGroupBox('Configuración de Filtros I:', self)
        self.gbx4.setGeometry(10, 200, 130, 220)
        self.vbox4 = QVBoxLayout(self)
        self.vbox4.addWidget(QLabel('Clasificador:', self))
        clasif_ipf = IPF.clasif()
        check3 = True
        self.RbClasif_ipf = []
        for s,cl in enumerate(clasif_ipf):
            if check3:
                butt = QRadioButton(str(cl[2]))
                butt.setChecked(True)
                self.vbox4.addWidget(butt)
                self.RbClasif_ipf.append(butt)
                check3 = False
            else:
                butt = QRadioButton(str(cl[2]))
                self.vbox4.addWidget(butt)
                self.RbClasif_ipf.append(butt)
        self.gbx4.setLayout(self.vbox4)
        self.gbx4.hide()
        
        self.gbx6 = QGroupBox('Configuración de Filtros II:', self)
        self.gbx6.setGeometry(145, 200, 140, 115)
        self.vbox6 = QVBoxLayout(self)
        self.vbox6.addWidget(QLabel('Cross-Validation:', self))
        sld2 = QSlider(Qt.Horizontal, self)
        sld2.setMinimum(2)
        sld2.setMaximum(10)
        sld2.setValue(5)
        sld2.setTickPosition(QSlider.TicksBelow)
        sld2.setTickInterval(1)
        self.num2 = QLCDNumber(self)
        self.num2.display(sld2.value())
        sld2.valueChanged.connect(self.num2.display)
        self.vbox6.addWidget(sld2)
        self.vbox6.addWidget(self.num2)
        self.gbx6.setLayout(self.vbox6)
        self.gbx6.hide()
        #Votos--------------------------------
        self.gbx5 = QGroupBox('Configuración de Filtros III:', self)
        self.gbx5.setGeometry(145, 320, 140, 100)
        self.vbox5 = QVBoxLayout(self)
        self.vbox5.addWidget(QLabel('Votación:', self))
        self.RbVotacion = []
        butt2 = QRadioButton("Consenso")
        butt2.setChecked(True)
        self.vbox5.addWidget(butt2)
        butt2_b = QRadioButton("Mayoria")
        self.vbox5.addWidget(butt2_b)
        self.RbVotacion.append(butt2)
        self.RbVotacion.append(butt2_b)
        self.gbx5.setLayout(self.vbox5)
        self.gbx5.hide()
        #EF----------------------------
        self.gbx7 = QGroupBox('Configuración de Filtros IV:', self)
        self.gbx7.setGeometry(290, 200, 130, 220)
        self.vbox7 = QVBoxLayout(self)
        self.vbox7.addWidget(QLabel('Clasificador en filtro:', self))
        cla_F_EF = EF.cla_filter()
        nombre1 = "\n"
        self.RbClasif_F_ef = []
        for s,cl in enumerate(cla_F_EF):
            nombre1 = nombre1 + str(cl[2]) + "\n"
        cla_F2_EF = EF.cla_filter2()
        nombre2 = "\n"
        for s,cl in enumerate(cla_F2_EF):
            nombre2 = nombre2 + str(cl[2]) + "\n"
        butt3 = QRadioButton(nombre1)
        butt3.setChecked(True)
        self.vbox7.addWidget(butt3)
        butt3_b = QRadioButton(nombre2)
        self.vbox7.addWidget(butt3_b)
        self.RbClasif_F_ef.append(butt3)
        self.RbClasif_F_ef.append(butt3_b)
        self.gbx7.setLayout(self.vbox7)
        self.gbx7.hide()
        #CVCF----------------------------
        self.gbx8 = QGroupBox('Configuración de Filtros IV:', self)
        self.gbx8.setGeometry(290, 200, 130, 220)
        self.vbox8 = QVBoxLayout(self)
        self.vbox8.addWidget(QLabel('Clasificador en filtro:', self))
        cla_F_CVCF = CVCF.cla_filter_cvcf()
        check4 = True
        self.RbClasif_F_cvcf = []
        for s,cl in enumerate(cla_F_CVCF):
            if check4:
                butt = QRadioButton(str(cl[2]))
                butt.setChecked(True)
                self.vbox8.addWidget(butt)
                self.RbClasif_F_cvcf.append(butt)
                check4 = False
            else:
                butt = QRadioButton(str(cl[2]))
                self.vbox8.addWidget(butt)
                self.RbClasif_F_cvcf.append(butt)
        self.gbx8.setLayout(self.vbox8)
        self.gbx8.hide()
        #IPF----------------------------
        self.gbx9 = QGroupBox('Configuración de Filtros IV:', self)
        self.gbx9.setGeometry(290, 200, 130, 220)
        self.vbox9 = QVBoxLayout(self)
        self.vbox9.addWidget(QLabel('Clasificador en filtro:', self))
        cla_F_IPF = IPF.cla_filter_ipf()
        check5 = True
        self.RbClasif_F_ipf = []
        for s,cl in enumerate(cla_F_IPF):
            if check5:
                butt = QRadioButton(str(cl[2]))
                butt.setChecked(True)
                self.vbox9.addWidget(butt)
                self.RbClasif_F_ipf.append(butt)
                check5 = False
            else:
                butt = QRadioButton(str(cl[2]))
                self.vbox9.addWidget(butt)
                self.RbClasif_F_ipf.append(butt)
        self.gbx9.setLayout(self.vbox9)
        self.gbx9.hide()
    def btnstate(self,b):
        
        if b.text() == "Ensemble Filter":
            if b.isChecked() == True:
                self.gbx2.show()
                self.gbx5.show()
                self.gbx6.show()
                self.gbx7.show()
                self.btn1.show()
#                print (b.text()+" is selected")
            else:
#                print (b.text()+" is deselected")
                self.gbx2.hide()
                self.gbx7.hide()		
        if b.text() == "Cross-Validation Committees Filter":
            if b.isChecked() == True:
                self.gbx3.show()
                self.gbx5.show()
                self.gbx6.show()
                self.gbx8.show()
                self.btn1.show()
#                print (b.text()+" is selected") 
            else:
#                print (b.text()+" is deselected")
                self.gbx3.hide()
                self.gbx8.hide()
        if b.text() == "Iterative Partitioning Filter":
            if b.isChecked() == True:
                self.gbx4.show()
                self.gbx5.show()
                self.gbx6.show()
                self.gbx9.show()
                self.btn1.show()
#                print (b.text()+" is selected")
            else:
#                print (b.text()+" is deselected")
                self.gbx4.hide()
                self.gbx9.hide()
    def addSliderRuido(self):
        self.gbx10 = QGroupBox('Ruido:', self)
        self.gbx10.setGeometry(250, 95, 170, 100)
        self.vbox10 = QVBoxLayout(self)
        sld = QSlider(Qt.Horizontal, self)
        sld.setMinimum(0)
        sld.setMaximum(6)
        sld.setValue(0)
        sld.setSingleStep(1)
        sld.setPageStep(1)
        sld.setTickPosition(QSlider.TicksBelow)
        sld.setTickInterval(1)
        self.num = QLCDNumber(self)
        self.num.display(sld.value())
        sld.valueChanged.connect(self.valChange)
        self.vbox10.addWidget(sld)
        self.vbox10.addWidget(self.num)
        self.gbx10.setLayout(self.vbox10)
        
    def valChange(self,value):
        value = value * 5
        self.num.display(value)

    def ls(self,ruta):
        dir, subdirs, archivos = next(walk(ruta))

        try:
            subdirs.remove("__pycache__")
        except:
            print()
            
        return subdirs

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ejm = FilterTFG()
    ejm.show()
    app.aboutToQuit.connect(app.deleteLater)
    sys.exit(app.exec_())