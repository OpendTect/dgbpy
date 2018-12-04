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
                             QFileDialog,
                             QFrame, QFormLayout, QHBoxLayout, QLineEdit,
                             QPushButton, QSizePolicy,
                             QSpinBox, QVBoxLayout, QWidget)

def getSpinBox(min,max,defval):
  ret = QSpinBox()
  ret.setRange(min,max)
  ret.setValue(defval)
  ret.setSizePolicy( QSizePolicy.Fixed, QSizePolicy.Preferred )
  return ret

def getFileInput(filenm):
  ret = QHBoxLayout()
  inpfilefld = QLineEdit(filenm)
  inpfilefld.setMinimumWidth(200)
  fileselbut = QPushButton('Select')
  ret.addWidget(inpfilefld)
  ret.addWidget(fileselbut)
  return (ret, inpfilefld, fileselbut)

def selectInput(parent,dlgtitle,dirnm,filters,lineoutfld):
  newfilenm = QFileDialog.getOpenFileName(parent,dlgtitle,dirnm,filters)
  lineoutfld.setText( newfilenm[0] )

def addSeparator(layout):
  linesep = QFrame()
  linesep.setFrameShape(QFrame.HLine)
  linesep.setFrameShadow(QFrame.Raised)
  layout.addWidget( linesep )

class WidgetGallery(QDialog):
  def __init__(self, args, parent=None):
    super(WidgetGallery, self).__init__(parent)
    self.lm = odcommon.LogManager( args )
    self.h5filenm = args['h5file'].name

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
    (self.inputfld, self.filenmfld, self.fileselbut) = getFileInput( self.h5filenm )
    layout.addRow( "&Train data", self.inputfld )
    layout.labelForField(self.inputfld).setBuddy(self.fileselbut)
    self.fileselbut.clicked.connect(lambda: selectInput(self,"Select training dataset",
                                                os.path.dirname( self.filenmfld.text() ),
                                                "HDF5 Files (*.hdf5 *.h5)",
                                                self.filenmfld) )

    if self.lm.log_file.name != '<stdout>':
      (self.logfld, self.lognmfld, self.logselbut) = getFileInput( self.lm.log_file.name )
      layout.addRow( "&Log File", self.logfld )
      layout.labelForField(self.logfld).setBuddy(self.logselbut)
      self.logselbut.clicked.connect(lambda: selectInput(self,"Select Log File",
                                                os.path.dirname( self.lm.log_file.name ),
                                                "Log Files (*.txt *.TXT *.log)",
                                                self.lognmfld) )

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
    self.modelfld = QLineEdit("<new model>")
    layout.addRow( "&Output model", self.modelfld )

  def createButtonsBox(self):
    buttons = QDialogButtonBox()
    self.runbutton =  buttons.addButton( QDialogButtonBox.Apply )
    self.runbutton.setText("Run")
    self.closebutton = buttons.addButton( QDialogButtonBox.Close )
    self.runbutton.clicked.connect(self.doApply)
    self.closebutton.clicked.connect(self.reject)

    self.buttonslayout = QVBoxLayout()
    self.buttonslayout.addWidget( buttons )

  def getParams(self):
    ret = {
      'num_tot_iterations': self.iterfld.value(),
      'epochs': self.epochfld.value(),
      'batch_size': self.batchfld.value(),
      'opt_patience': self.patiencefld.value()
    }
    if self.dodecimate.isChecked():
      ret.update({'num_train_ex': self.decimatefld.value()})
    return ret

  def doApply(self):
    params = self.getParams();
    self.lm.log_msg( "Input: " + os.path.basename(self.filenmfld.text()) )
    if self.lognmfld != None:
      self.lm.log_msg( "Log: " + os.path.basename(self.lognmfld.text()) )
    self.lm.log_msg( params )


def setStyleSheet( app ):
  cssfile = open( "/auto/d29/arnaud/dev/od/data/Styles/default.qss", "r" )
  qtcss = cssfile.read()
  cssfile.close()
  app.setStyleSheet( qtcss )


if __name__ == '__main__':

    import sys
    import os
    import argparse
    import signal
    import odpy.common as odcommon

    parser = argparse.ArgumentParser(prog='PROG',description='Select parameters for training a Keras model')
    parser.add_argument('-v','--version',action='version',version='%(prog)s 1.0')
    parser.add_argument('h5file',type=argparse.FileType('r'),
                        help='HDF5 file containing the training data')
    parser.add_argument('--log',dest='logfile',metavar='file',nargs='?',
                        type=argparse.FileType('a'),
                        default='sys.stdout',help='Progress report output')
    parser.add_argument('--syslog',dest='sysout',metavar='stdout',nargs='?',
                        type=argparse.FileType('a'),
                        default='sys.stdout',help='Standard output')
    args = vars(parser.parse_args())

    app = QApplication(sys.argv)
    gallery = WidgetGallery(args)
    gallery.show()
    setStyleSheet( app )

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_()) 
