#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Bert
# DATE     : August 2018
#
# Script for deep learning train
# is called by the DeepLearning plugin
#


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QFrame, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QLayoutItem, QLineEdit, QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)

def getSpinBox(grp,min,max,defval):
  ret = QSpinBox(grp)
  ret.setRange(min,max)
  ret.setValue(defval)
  return ret

def addSeparator(layout):
  linesep = QFrame()
  linesep.setFrameShape(QFrame.HLine)
  linesep.setFrameShadow(QFrame.Raised)
  layout.addWidget( linesep )

def getMaxLabelWidth(formlayout):
  ret = 0
  for row in range(formlayout.rowCount()):
    widgetitm = formlayout.itemAt(row,QFormLayout.LabelRole)
    widwidth = 0
    if widgetitm.isEmpty():
      widwidth = widgetitm.widget().width()
    else:
      widwidth = widgetitm.widget().width()
    if widwidth > ret:
      ret = widwidth
  return ret

class WidgetGallery(QDialog):
  def __init__(self, parent=None):
    super(WidgetGallery, self).__init__(parent)

    self.createInputGroupBox()
    self.createParametersGroupBox()
    self.createOutputGroupBox()

    mainLayout = QVBoxLayout()
    mainLayout.addLayout( self.inplayout )
    addSeparator( mainLayout  )
    mainLayout.addLayout( self.paramslayout )
    addSeparator( mainLayout  )
    mainLayout.addLayout( self.outlayout )
    #mainLayout.addWidget(self.inputgrp)
    #mainLayout.addWidget(self.paramgrp)
    #mainLayout.addWidget(self.outputgrp)
    self.setLayout(mainLayout)

  def createInputGroupBox(self):
    self.inputgrp = QGroupBox("Input")

    self.inputfld = QLabel('path-to-h5file.h5',self.inputgrp)
    self.inplogfld = QLabel('path-to-logfile.txt',self.inputgrp)

    self.inplayout = QFormLayout()
    self.inplayout.setLabelAlignment( Qt.AlignRight )
    self.inplayout.addRow( "&Train data", self.inputfld )
    self.inplayout.addRow( "&Log File", self.inplogfld )
    #self.inputgrp.setLayout( self.inplayout )

  def createParametersGroupBox(self):
    self.paramgrp = QGroupBox("Keras training parameters")

    self.iterfld = getSpinBox(self.paramgrp,1,100,15)
    self.epochfld = getSpinBox(self.paramgrp,1,1000,15)
    self.dodecimate = QCheckBox( "&Decimate input", self.paramgrp )
    self.dodecimate.setTristate( False )
    self.dodecimate.setChecked( False )
    self.decimatefld = getSpinBox(self.paramgrp,1,99,10)

    self.batchfld = getSpinBox(self.paramgrp,1,1000,16)
    self.patiencefld = getSpinBox(self.paramgrp,1,1000,10)

    self.paramslayout = QFormLayout()
    self.paramslayout.setLabelAlignment( Qt.AlignRight )
    self.paramslayout.addRow( "Number of &Iterations", self.iterfld )
    self.paramslayout.addRow( "Number of &Epochs", self.epochfld )
    self.paramslayout.addRow( self.dodecimate, self.decimatefld )
    self.paramslayout.addRow( "Number of &Batch", self.batchfld )
    self.paramslayout.addRow( "&Patience", self.patiencefld )
    #self.paramgrp.setLayout( self.paramslayout )

  def createOutputGroupBox(self):
    self.outputgrp = QGroupBox("Output")

    modellbl = QLabel("&Trained model",self.outputgrp)
    self.modelfld = QLabel("Trained HDF5 model",self.outputgrp)
    modellbl.setBuddy(self.modelfld)

    self.outlayout = QFormLayout()
    self.outlayout.setLabelAlignment( Qt.AlignRight )
    self.outlayout.addRow( modellbl, self.modelfld )
    #self.outputgrp.setLayout(self.outlayout)

  def printSummary(self,lm):
    decim = None
    if self.dodecimate.isChecked():
      decim = self.decimatefld.value()
    ret = {
      'num_tot_iterations': self.iterfld.value(),
      'epochs': self.epochfld.value(),
      'num_train_ex': decim,
      'batch_size': self.batchfld.value(),
      'opt_patience': self.patiencefld.value()
    }
    lm.log_msg( ret )


if __name__ == '__main__':

    import os
    import sys
    import argparse
    import odpy.common as odcommon

    def setStyleSheet( app ):
      cssfile = open( "/auto/d29/arnaud/dev/od/data/Styles/default.qss", "r" )
      qtcss = cssfile.read()
      cssfile.close()
      app.setStyleSheet( qtcss )

    parser = argparse.ArgumentParser(prog='PROG',description='Select parameters for training a Keras model')
    parser.add_argument('-v','--version',action='version',version='%(prog)s 1.0')
    parser.add_argument('--log',dest='logfile',metavar='file',nargs='?',type=argparse.FileType('a'),
                        default='sys.stdout',help='Progress report output')
    parser.add_argument('--syslog',dest='sysout',metavar='stdout',nargs='?',type=argparse.FileType('a'),
                        default='sys.stdout',help='Standard output')
    args = vars(parser.parse_args())
    lm = odcommon.LogManager( args )

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    setStyleSheet( app )
    ret = app.exec_() 
    gallery.printSummary( lm )

    sys.exit(ret) 
