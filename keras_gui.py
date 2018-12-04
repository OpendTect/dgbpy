#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud Huck
# DATE     : December 2018
#
# Script for deep learning train
# is called by the DeepLearning plugin
#


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
                             QFrame, QFormLayout,QLabel, QSpinBox, QVBoxLayout,
                             QWidget)

def getSpinBox(min,max,defval):
  ret = QSpinBox()
  ret.setRange(min,max)
  ret.setValue(defval)
  return ret

def addSeparator(layout):
  linesep = QFrame()
  linesep.setFrameShape(QFrame.HLine)
  linesep.setFrameShadow(QFrame.Raised)
  layout.addWidget( linesep )

class WidgetGallery(QDialog):
  def __init__(self, args, parent=None):
    super(WidgetGallery, self).__init__(parent)
    self.lm = odcommon.LogManager( args )

    mainform = QFormLayout()
    mainform.setLabelAlignment( Qt.AlignRight )
    self.createInputGroupBox( mainform )
    self.createParametersGroupBox( mainform )
    self.createOutputGroupBox( mainform )
    self.createButtonsBox()

    mainlayout = QVBoxLayout()
    mainlayout.addLayout( mainform )
    addSeparator( mainlayout  )
    mainlayout.addLayout( self.buttonslayout )
    self.setLayout( mainlayout )

  def createInputGroupBox(self,layout):
    self.inputfld = QLabel('path-to-h5file.h5')
    self.inplogfld = QLabel('path-to-logfile.txt')

    layout.addRow( "&Train data", self.inputfld )
    layout.addRow( "&Log File", self.inplogfld )

  def createParametersGroupBox(self,layout):
    self.iterfld = getSpinBox(1,100,15)
    self.epochfld = getSpinBox(1,1000,15)
    self.dodecimate = QCheckBox( "&Decimate input" )
    self.dodecimate.setTristate( False )
    self.dodecimate.setChecked( False )
    self.decimatefld = getSpinBox(1,99,10)
    self.decimatefld.setSuffix("%")
    self.decimatefld.setDisabled( True )
    self.dodecimate.toggled.connect(self.decimatefld.setEnabled)
    self.batchfld = getSpinBox(1,1000,16)
    self.patiencefld = getSpinBox(1,1000,10)

    layout.addRow( "Number of &Iterations", self.iterfld )
    layout.addRow( "Number of &Epochs", self.epochfld )
    layout.addRow( self.dodecimate, self.decimatefld )
    layout.addRow( "Number of &Batch", self.batchfld )
    layout.addRow( "&Patience", self.patiencefld )

  def createOutputGroupBox(self,layout):
    self.modelfld = QLabel("Trained HDF5 model")
    layout.addRow( "&Trained model", self.modelfld )

  def createButtonsBox(self):
    buttons = QDialogButtonBox()
    self.runbutton =  buttons.addButton( QDialogButtonBox.Apply )
    self.runbutton.setText("Run")
    self.closebutton = buttons.addButton( QDialogButtonBox.Close )
    self.runbutton.clicked.connect(self.doApply)
    self.closebutton.clicked.connect(self.reject)

    self.buttonslayout = QVBoxLayout()
    self.buttonslayout.addWidget( buttons )

  def doApply(self):
    self.printSummary();

  def printSummary(self):
    ret = {
      'num_tot_iterations': self.iterfld.value(),
      'epochs': self.epochfld.value(),
      'batch_size': self.batchfld.value(),
      'opt_patience': self.patiencefld.value()
    }
    if self.dodecimate.isChecked():
      ret.update({'num_train_ex': self.decimatefld.value()})
    self.lm.log_msg( ret )

def setStyleSheet( app ):
  cssfile = open( "/auto/d29/arnaud/dev/od/data/Styles/default.qss", "r" )
  qtcss = cssfile.read()
  cssfile.close()
  app.setStyleSheet( qtcss )


if __name__ == '__main__':

    import os
    import signal
    import sys
    import argparse
    import odpy.common as odcommon

    parser = argparse.ArgumentParser(prog='PROG',description='Select parameters for training a Keras model')
    parser.add_argument('-v','--version',action='version',version='%(prog)s 1.0')
    parser.add_argument('--log',dest='logfile',metavar='file',nargs='?',type=argparse.FileType('a'),
                        default='sys.stdout',help='Progress report output')
    parser.add_argument('--syslog',dest='sysout',metavar='stdout',nargs='?',type=argparse.FileType('a'),
                        default='sys.stdout',help='Standard output')
    args = vars(parser.parse_args())

    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    gallery = WidgetGallery(args)
    gallery.show()
    setStyleSheet( app )

    sys.exit(app.exec_()) 
