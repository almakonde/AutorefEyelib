import os
import sys
import dateutil
import re

import xml.etree.ElementTree as ET
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers.polling import PollingObserver # for network path
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRect
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtWidgets import QTabWidget
import pyperclip
from autorefeyelib.Parsers import vx120 as vxParser
from autorefeyelib.Refraction import Predictor as vxPredictor
from autorefeyelib.IOLpower.Predictor import Predictor as iolPredictor
from autorefeyelib.RefractiveSurgery import SurgeryRecommender 
from autorefeyelib.GUI.SubjRefraction import Tab as subjRefTab

# TODO: add logging
# TODO: add sorting options for patient list
# TODO: grey-out non editable fields
# TODO: add parameter json file
# TODO: change monitor folder if folder is not accessible
class GUI():

    def __enter__(self):
        print("[Info][GUI] Starting")
        return self

    def __exit__(self, type, value, traceback):
        # self._window.destroy()
        self.gui_main_window.closeAllWindows()
        self.gui_main_window.exit()
        self.gui_main_window.quit()
        print(f"[Info][GUI] Exporting prediction list to {self._predictedDB_path}")
        self._predictions.to_csv(self._predictedDB_path,
                                 encoding='utf-8',
                                 sep=',',
                                 index=True)
        print("[Info][GUI] Closing application")

    def __init__(self,data_path,monitorVxFolder=True,**kwargs):

        """ class constructor """
        # TODO: move parameters to json file
        self.data_path_          = data_path     # folder to monitor
        codeDir                  = os.path.dirname(__file__)
        # self.vx_db_path_         = os.path.join(codeDir,'..','..','data','vx130Data_Prevot.csv')
        self.models_folder_      = os.path.join(codeDir,'..','Refraction','models')
        # self._vx40_xml_path_     = os.path.join(codeDir,'..','etc','VX40_Sample.xml')
        self._predictedDB_path   = os.path.join(codeDir,'predicted.csv')
        self._xml_output_path    = os.path.join(codeDir,'Patients_xml')
        self._xml_emr_path       = os.path.join(codeDir,'patients_xml')
        self.monitorVxFolder     = monitorVxFolder
        self.currentPat          = pd.DataFrame()
        self.currentPrediction   = pd.DataFrame()
        self.vx_parser           = vxParser.Parser()
        self.vx_predictor        = vxPredictor.Predictor()
        self._tableEditCount     = 0
        self.iol_predictor       = iolPredictor()
        self.surgeryRecommender = SurgeryRecommender.Recommender()

        fieldNames01  = []#list(self.vx_parser.data.columns)
        fieldNames02  = ['Predicted_Sphere_Right','Predicted_Cylinder_Right',
                         'Predicted_Axis_Right','Predicted_Add_Right','Predicted_Sphere_Left',
                         'Predicted_Cylinder_Left','Predicted_Axis_Left','Predicted_Add_Left',
                         'Predicted_Contact_Sphere_Right','Predicted_Contact_Cylinder_Right',
                         'Predicted_Contact_Axis_Right','Predicted_Contact_Sphere_Left',
                         'Predicted_Contact_Cylinder_Left','Predicted_Contact_Axis_Left']
        self._columns_to_export_ = fieldNames01+fieldNames02
        # check for paths and files integrity
        self._validate_folders_and_files_integrity()
        # load previous predictions
        self._predictions = pd.read_csv(self._predictedDB_path,index_col=0)

        # self._vx40_xml    = ET.parse(self._vx40_xml_path_)

        file_list_      = os.listdir(self.data_path_)
        # keep only folders, filter out files
        ind             = 0
        sDate           = []
        fList           = []
        for fIdx in file_list_:
            if os.path.isdir(os.path.join(self.data_path_,fIdx)):
                sDate.append(os.path.getctime(os.path.join(self.data_path_,fIdx)))
                fList.append(fIdx)
            elif fIdx.endswith('zip'):
                sDate.append(os.path.getctime(os.path.join(self.data_path_,fIdx)))
                fList.append(fIdx)
            ind+=1

        # sortedInds  = sorted(range(len(sDate)), key=lambda k: sDate[k],reverse=True)
        sortedInds = sorted(range(len(sDate)),key=lambda k:sDate[k],reverse=True)
        fListSorted = fList.copy()
        for fIdx in range(len(fList)):
            fListSorted[fIdx] = fList[sortedInds[fIdx]]
        self.file_list_ = fListSorted

        self._ConstructFolderMonitor()
        self.construct_gui_main_window()

    def _validate_folders_and_files_integrity(self):

        # if os.path.exists(self.vx_db_path_):
        #     print("[Info][GUI] vx120 DB found")
        # else:
        #     print("[Warn][GUI] Cannot find vx db")

        if os.path.isdir(self.models_folder_):
            if os.path.exists(os.path.join(self.models_folder_,'CylModel.sav')):
                print("[Info][GUI] Found cylinder model")
            else:
                print(f"[Error][GUI] Cannot find CylModel.sav at {self.models_folder_} ")
            if os.path.exists(os.path.join(self.models_folder_,"SphModel.sav")):
                print("[Info][GUI] found sphere model")
            else:
                print(f"[Error][GUI] cannot find SphModel.sav at {self.models_folder_}")
        else:
            print(f"[Error][GUI] Cannot find {self.models_folder_}")

        # if os.path.exists(self._vx40_xml_path_):
        #     print("[Info][GUI] Found vx40 xml")
        # else:
        #     print(f"[Error][GUI] Cannot find {self._vx40_xml_path_}")

        if os.path.exists(self._predictedDB_path):
            print("[Info][GUI] Found predicted DB")
        else:
            print(f"[Warn][GUI] Cannot find {self._predictedDB_path}. Generating a new one")
            pd.DataFrame(columns = self._columns_to_export_).to_csv(self._predictedDB_path,index=True)

        if os.path.isdir(self._xml_output_path):
            print(f"[Info][GUI] found {self._xml_output_path}")
        else:
            print(f"[Warn][GUI] cannot find {self._xml_output_path} Creating it")
            os.mkdir(self._xml_output_path)

    def construct_gui_main_window(self):
        """Construct all GUI widgets  """
        # application
        print("[Info][GUI] Creating main window")
        self.gui_main_window = QtWidgets.QApplication([])
        with open(os.path.join(os.path.dirname(__file__),"gui_style.qss"), "r") as f:
            _style = f.read()
            self.gui_main_window.setStyleSheet(_style)
        self.gui_main_window.setDesktopFileName('AutoRefeaction')
        self.gui_main_window.setApplicationName('Subjective refraction powered by machine learning')
        self.gui_main_window.setObjectName('AutoRefraction')

        # self.gui_main_window.setWindowIcon(QtGui.QIcon(os.path.join(os.getcwd(),'..','etc','mikajaki_logo.ico')))
        # main widget
        self._window = QtWidgets.QWidget()
        self._window.resize(600,900)
        
        # define tabs
        self._tabs = QTabWidget(self._window)
        self._tab_subj_pred = QtWidgets.QWidget()
        self._tab_vx_images = QtWidgets.QWidget()
        self._tabs.resize(600,900)
        self._tabs.addTab(self._tab_subj_pred,"Subj. Ref.")
        self._tabs.addTab(self._tab_vx_images,"Surg Recom.")

        # Add file (patient) list
        QtWidgets.QLabel(self._tab_subj_pred,text="Patients",geometry=QRect(20,10,50,30))
        self._patient_list = QtWidgets.QListWidget(self._tab_subj_pred,geometry=QRect(20, 40, 260, 490))
        #TODO: allow patient sorting 
        for lIdx in range(len(self.file_list_)):
            fName = self.file_list_[lIdx].replace("#"," ").replace('export_patient_','').replace('_',' ').replace('.zip','')
            self._patient_list.insertItem(lIdx,fName)

        self._patient_list.itemActivated.connect(self._list_select_callback)
        self._patient_list.alternatingRowColors = False
        self._patient_list.dragDropMode         = False

        # Add patient info panel to display prediction
        self._patient_info = QtWidgets.QTableWidget(self._tab_subj_pred,
                             geometry = QRect(290, 40, 270, 150),
                             columnCount=1,
                             rowCount=4)
        self._patient_info.setColumnWidth(0,200)
        for i in range(self._patient_info.rowCount()):
         for j in range(self._patient_info.columnCount()):
             item = QtWidgets.QTableWidgetItem('')
             item.setFlags(QtCore.Qt.ItemIsEnabled)
             self._patient_info.setItem(i,j,item)
        self._patient_info.setVerticalHeaderLabels(['Surname','Name','Gender','Age'])
        self._patient_info.setHorizontalHeaderLabels([""])
        # self._patient_info.setText("Patient Info")

        tableHeaderNames = ['Sphere','Cylinder','Axis','SE','ACD','WTW','K1','K2','Pachy','AL','Rt','Aconst']
        tableParamName   = ['WF_SPHERE_R_3',
                            'WF_CYLINDER_R_3',
                            'WF_AXIS_R_3',
                            'SphericalEquivalent_3',
                            'Pachy_MEASURE_Acd',
                            'Pachy_MEASURE_WhiteToWhite',
                            'Topo_Sim_K_K1',
                            'Topo_Sim_K_K2',
                            'Pachy_MEASURE_Thickness',
                            'AxialLength',
                            'targetRefraction',
                            'Aconst']

        self._patient_info_table = QtWidgets.QTableWidget(self._tab_subj_pred,
                                    geometry= QRect(290, 200, 280, 330),
                                    columnCount=2,
                                    accessibleName="patientInfoTable",
                                    rowCount=len(tableHeaderNames))
        self._patient_info_table.setHorizontalHeaderLabels(['Right','Left'])
        self._patient_info_table.setVerticalHeaderLabels(tableHeaderNames)
        self._patient_info_table.itemChanged.connect(self._patient_table_edit_callback)
        # make table read only
        eInd = ['_Right','_Left']
        for i in range(self._patient_info_table.rowCount()):
            for j in range(self._patient_info_table.columnCount()):
                item = QtWidgets.QTableWidgetItem('')
                item.setWhatsThis(tableParamName[i]+eInd[j])
                # item.setFlags(QtCore.Qt.ItemIsEnabled)
                self._patient_info_table.setItem(i,j,item)

        self._editGroupBox = QtWidgets.QGroupBox(self._tab_subj_pred)
        self._editGroupBox.setGeometry(QRect(20, 550, 550, 300))

        # Add edit boxes for Sph Cyl and Ax (right and left)
        QtWidgets.QLabel(self._editGroupBox,
                        text='Glasses',
                        geometry=QRect(10,30,120,30))

        self._editGroupBox.setTitle('Machine Learning Prediction')

        # accessible names of widgets should match the field names for predicted component
        # e.g Predicted_Sphere_Left
        self._rSph_edit = QtWidgets.QLineEdit(self._editGroupBox,
                geometry        = QRect(10,60,50,30),
                placeholderText = "sph",
                accessibleName  = "Predicted_Sphere_Right",
                alignment       = QtCore.Qt.AlignCenter,
                maxLength       = 5)
        self._rSph_edit.editingFinished.connect(self._edit_refraction_callback)

        self._rCyl_edit = QtWidgets.QLineEdit(self._editGroupBox,
                geometry        = QRect(60,60,50,30),
                accessibleName  = "Predicted_Cylinder_Right",
                placeholderText = "Cyl",
                alignment       = QtCore.Qt.AlignCenter,
                maxLength       = 5)
        self._rCyl_edit.editingFinished.connect(self._edit_refraction_callback)

        self._rAx_edit = QtWidgets.QLineEdit(self._editGroupBox,
                geometry        = QRect(110,60,50,30),
                accessibleName  = "Predicted_Axis_Right",
                placeholderText = "Ax",
                alignment       = QtCore.Qt.AlignCenter,
                maxLength       = 3)
        self._rAx_edit.editingFinished.connect(self._edit_refraction_callback)

        QtWidgets.QLabel(self._editGroupBox,text="Add:",
                        geometry=QRect(160,60,30,30))
        self._rAdd_edit = QtWidgets.QLineEdit(self._editGroupBox,
                    geometry        = QRect(190,60,50,30),
                    accessibleName  = "Predicted_Add_Right",
                    alignment       = QtCore.Qt.AlignCenter,
                    maxLength       = 5)
        self._rAdd_edit.editingFinished.connect(self._edit_refraction_callback)

        QtWidgets.QLabel(self._editGroupBox,text="/",geometry=QRect(240,60,10,30))

        self._lSph_edit = QtWidgets.QLineEdit(self._editGroupBox,
                        geometry        = QRect(250,60,50,30),
                        accessibleName  = "Predicted_Sphere_Left",
                        placeholderText = "Sph",
                        alignment       = QtCore.Qt.AlignCenter,
                        maxLength       = 5)
        self._lSph_edit.editingFinished.connect(self._edit_refraction_callback)

        self._lCyl_edit = QtWidgets.QLineEdit(self._editGroupBox,
                geometry        = QRect(300,60,50,30),
                placeholderText = "Cyl",
                accessibleName  = "Predicted_Cylinder_Left",
                alignment       = QtCore.Qt.AlignCenter,
                maxLength       = 5)

        self._lCyl_edit.editingFinished.connect(self._edit_refraction_callback)

        self._lAx_edit  = QtWidgets.QLineEdit(self._editGroupBox,
                geometry        = QRect(350,60,50,30),
                placeholderText = "Ax",
                accessibleName  = "Predicted_Axis_Left",
                # inputMask       = "000",
                alignment       = QtCore.Qt.AlignCenter,
                maxLength       = 3)
        self._lAx_edit.editingFinished.connect(self._edit_refraction_callback)

        QtWidgets.QLabel(self._editGroupBox,text="Add:",geometry=QRect(400,60,30,30))
        self._lAdd_edit = QtWidgets.QLineEdit(self._editGroupBox,
                                              geometry       = QRect(430,60,50,30),
                                              accessibleName = "Predicted_Add_Left",
                                              alignment      = QtCore.Qt.AlignCenter)
        self._lAdd_edit.editingFinished.connect(self._edit_refraction_callback)

        # create copy refraction button
        self._copy_ref = QtWidgets.QPushButton(self._editGroupBox,
                            text="copy",
                            geometry = QRect(500,60,50,30))
        self._copy_ref.clicked.connect(self._copy_ref_callback)

        # contact lenses
        QtWidgets.QLabel(self._editGroupBox,
                         text ='Contact lenses',
                         geometry=QRect(10,100,120,30))

        self._rcSph = QtWidgets.QLineEdit(self._editGroupBox,
                    geometry        = QRect(10,130,50,30),
                    readOnly        = True,
                    alignment       = QtCore.Qt.AlignCenter,
                    accessibleName  = "Predicted_Contact_Sphere_Right",
                    placeholderText = "Sph")

        self._rcCyl = QtWidgets.QLineEdit(self._editGroupBox,
                    geometry        = QRect(60,130,50,30),
                    readOnly        = True,
                    alignment       = QtCore.Qt.AlignCenter,
                    accessibleName  = "Predicted_Contact_Cylinder_Right",
                    placeholderText = "Cyl")

        self._rcAx = QtWidgets.QLineEdit(self._editGroupBox,
                    geometry        = QRect(110,130,50,30),
                    readOnly        = True,
                    alignment       = QtCore.Qt.AlignCenter,
                    accessibleName  = "Predicted_Contact_Axis_Right",
                    placeholderText = "Ax")

        QtWidgets.QLabel(self._editGroupBox,
                        text="/",
                        geometry=QRect(160,130,10,30))
        self._lcSph = QtWidgets.QLineEdit(self._editGroupBox,
                     geometry        = QRect(170,130,50,30),
                     readOnly        = True,
                     alignment       = QtCore.Qt.AlignCenter,
                     accessibleName  = "Predicted_Contact_Sphere_Left",
                     placeholderText = "Sph")

        self._lcCyl = QtWidgets.QLineEdit(self._editGroupBox,
                     geometry        = QRect(220,130,50,30),
                     readOnly        = True,
                     alignment       = QtCore.Qt.AlignCenter,
                     accessibleName  = "Predicted_Contact_Cylinder_Left",
                     placeholderText = "Cyl")

        self._lcAx = QtWidgets.QLineEdit(self._editGroupBox,
                     geometry        = QRect(270,130,50,30),
                     readOnly        = True,
                     alignment       = QtCore.Qt.AlignCenter,
                     accessibleName  = "Predicted_Contact_Axis_Left",
                     placeholderText = "Ax")

        self._copy_contact = QtWidgets.QPushButton(self._editGroupBox,
                            text="copy",
                            geometry = QRect(500,130,50,30))
        self._copy_contact.clicked.connect(self._copy_contact_callback)

        QtWidgets.QLabel(self._editGroupBox,
                         text ='IOL',
                         geometry=QRect(10,170,120,40))

        QtWidgets.QLabel(self._editGroupBox,
                         text ='Power',
                         geometry=QRect(10,210,120,40))

        QtWidgets.QLabel(self._editGroupBox,
                         text ='Refraction',
                         geometry=QRect(10,260,120,40))

        self._rIOL = QtWidgets.QLineEdit(self._editGroupBox,
                     geometry        = QRect(80,210,50,30),
                     readOnly        = True,
                     alignment       = QtCore.Qt.AlignCenter,
                     accessibleName  = "Predicted_IOL_Right",
                     placeholderText = "")

        self._rIolRef = QtWidgets.QLineEdit(self._editGroupBox,
                     geometry        = QRect(80,260,50,30),
                     readOnly        = True,
                     alignment       = QtCore.Qt.AlignCenter,
                     accessibleName  = "Predicted_IOL_Postop_Refraction_Right",
                     placeholderText = "")

        self._lIOL = QtWidgets.QLineEdit(self._editGroupBox,
                     geometry        = QRect(150,210,50,30),
                     readOnly        = True,
                     alignment       = QtCore.Qt.AlignCenter,
                     accessibleName  = "Predicted_IOL_Left",
                     placeholderText = "")

        self._lIolRef = QtWidgets.QLineEdit(self._editGroupBox,
                     geometry        = QRect(150,260,50,30),
                     readOnly        = True,
                     alignment       = QtCore.Qt.AlignCenter,
                     accessibleName  = "Predicted_IOL_Postop_Refraction_Left",
                     placeholderText = "")


        self._exportBtn = QtWidgets.QPushButton(self._editGroupBox,
                            text="Export xml to EMR",
                            geometry = QRect(400,250,150,30))
        self._exportBtn.clicked.connect(self._export_btn_callback)

        self._err_dialog = QtWidgets.QErrorMessage()
        self._window.show()

        # check the validity of all strings in edit boxes

    def _edit_refraction_callback(self,*args):

        flag1 = self._check_value_validity(self._rSph_edit)
        if flag1:
            val = float(self._rSph_edit.text())
            self.currentPrediction[self._rSph_edit.accessibleName()] = val
        else:
            self._err_dialog.showMessage("Wrong format of sphere value")
            self._err_dialog.exec()
            self._rSph_edit.setFocus()

        flag2 = self._check_value_validity(self._rCyl_edit)
        if flag2:
            val = float(self._rCyl_edit.text())
            if val<=0:
                self.currentPrediction[self._rCyl_edit.accessibleName()] = val
            else:
                self._err_dialog.showMessage("Cylinder must be negative")
                self._err_dialog.exec()
                self._rCyl_edit.setFocus()
                flag2 = False
        else:
            self._err_dialog.showMessage("Wrong format of cylinder value")
            self._err_dialog.exec()
            self._rCyl_edit.setFocus()

        flag3 = self._check_value_validity(self._rAx_edit)
        if flag3:
            val = float(self._rAx_edit.text())
            if val>=0 and val<=180:
                self.currentPrediction[self._rAx_edit.accessibleName()] = val
            else:
                self._err_dialog.showMessage("axis must be in the range 0-180")
                self._err_dialog.exec()
                self._rAdd_edit.setFocus()
                flag3 = False
        else:
            self._err_dialog.showMessage("Wrong format of axis value")
            self._err_dialog.exec()
            self._rAdd_edit.setFocus()

        flag4 = self._check_value_validity(self._lSph_edit)
        if flag4:
            val = float(self._lSph_edit.text())
            self.currentPrediction[self._lSph_edit.accessibleName()] = val

        flag5 = self._check_value_validity(self._lCyl_edit)
        if flag5:
            val = float(self._lCyl_edit.text())
            if val<=0:
                self.currentPrediction[self._lCyl_edit.accessibleName()] = val
            else:
                self._err_dialog.showMessage("Cylinder must be negative")
                self._err_dialog.exec()
                self._lCyl_edit.setFocus()
                # set focus back to the edit box
                flag5 = False
        else:
            self._err_dialog.showMessage("wrong format of cylinder value")
            self._err_dialog.exec()
            self._lCyl_edit.setFocus()

        flag6 = self._check_value_validity(self._lAx_edit)
        if flag6:
            val = float(self._lAx_edit.text())
            if val>=0 and val<=180:
                self.currentPrediction[self._lAx_edit.accessibleName()]= val
            else:
                self._err_dialog.showMessage("Axis must be in the range 0-180")
                self._err_dialog.exec()
                self._lAx_edit.setFocus()
                flag6 = False
        else:
            self._err_dialog.showMessage("Wrong format of axis value")
            self._err_dialog.exec()
            self._lAx_edit.setFocus()

        flag7 = self._check_value_validity(self._rAdd_edit)
        if flag7:
            val = float(self._rAdd_edit.text())
            if val>=0:
                self.currentPrediction[self._rAdd_edit.accessibleName()] = val
            else:
                self._err_dialog.showMessage("Addition must be positive")
                self._err_dialog.exec()
                self._rAdd_edit.setFocus()
                flag7 = False
        else:
            self._err_dialog.showMessage("Wrong format of addition value")
            self._err_dialog.exec()
            self._rAdd_edit.setFocus()

        flag8 = self._check_value_validity(self._lAdd_edit)
        if flag8:
            val = float(self._lAdd_edit.text())
            if val>=0:
                self.currentPrediction[self._lAdd_edit.accessibleName()] = val
            else:
                self._err_dialog.showMessage("Addition must be positive")
                self._err_dialog.exec()
                self._lAx_edit.setFocus()
                flag8 = False
        else:
            self._err_dialog.showMessage("Wrong format of addition value")
            self._err_dialog.exec()
            self._lAx_edit.setFocus()

        if flag1&flag2&flag3&flag4&flag5&flag6&flag7&flag8:
            self.vx_predictor._PredictiContactLenses(self.currentPrediction)
            self._display_prediction(self.currentPrediction)

    def _export_btn_callback(self,*args):
        """ Export results as vx40 xml (Visionix) """
        if len(self.currentPrediction)>0:
            self.vx_predictor.ExportPredictionToVx40xml(self.currentPrediction,self._xml_emr_path)

    def _list_item_change_callback(self,*args):
        """
         A callback for  Accept button press aignal
         if patient exist in the predictions list, replace it
         otherwise, append to prediction list
        """
        if len(self.currentPrediction)>0:
            # self._predictions.loc[self.currentPrediction.index] = self.currentPrediction
            patInds = self._predictions.index.values==self.currentPrediction.index.values
            if any(patInds):
                # for existing patient, overwrite patient data
                self._predictions.loc[self.currentPrediction.index,:] = self.currentPrediction
            else:
                # new patient
                self._predictions = self._predictions.append(self.currentPrediction)
                # self._predictions.reset_index()

            self.vx_predictor.ExportPredictionToVx40xml(self.currentPrediction,self._xml_output_path)

    def _copy_ref_callback(self,*args):
        """ Copy spectacle refraction to clipboard"""
        cp = self._format_current_prediction_string(rtype="spectacles")
        pyperclip.copy(cp)

    def _copy_contact_callback(self,*args):
        """ Copy contact refractin to clipboard """
        cp = self._format_current_prediction_string(rtype="contacts")
        pyperclip.copy(cp)

    def _patient_table_edit_callback(self,tableItem,*args):
        """ emitted whenever objective data has changed"""

        if len(self.currentPrediction)>0:
            paramName = tableItem.whatsThis()
            self.currentPrediction[paramName] = float(tableItem.text())
            self.currentPat[paramName]        = float(tableItem.text())
            self.currentPrediction = self.Predict(self.currentPat)
            self._display_prediction(self.currentPrediction)

    def _format_current_prediction_string(self,rtype="spectacles"):
        """
          Format the predicted refraction as a string

         Parameters:
         -----------
          type, str, default="spectacles"
           options: spectacles, contacts
        """
        if len(self.currentPrediction)!=0:
            cp = self.currentPrediction
            ds = u'\N{DEGREE SIGN}'
            if rtype=="spectacles":
                refStr = f"{cp['Predicted_Sphere_Right'][0]:.2f}"\
                       + f"({cp['Predicted_Cylinder_Right'][0]:.2f})"\
                       + f"{cp['Predicted_Axis_Right'][0]:.0f}"+ds \
                       +"/"\
                       + f"{cp['Predicted_Sphere_Left'][0]:.2f}"\
                       + f"({cp['Predicted_Cylinder_Left'][0]:.2f})"\
                       + f"{cp['Predicted_Axis_Left'][0]:.0f}"+ds \
                       + "*A"\
                       + f"{cp['Predicted_Add_Left'][0]:.2f}"
            elif rtype =="contacts":
                refStr = f"{cp['Predicted_Contact_Sphere_Right'][0]:.2f}"\
                       + f"({cp['Predicted_Contact_Cylinder_Right'][0]:.2f})"\
                       + f"{cp['Predicted_Contact_Axis_Right'][0]:.0f}"+ds \
                       + "/"\
                       + f"{cp['Predicted_Contact_Sphere_Left'][0]:.2f}"\
                       + f"({cp['Predicted_Contact_Cylinder_Left'][0]:.2f})"\
                       + f"{cp['Predicted_Contact_Axis_Left'][0]:.0f}"+ds
            else:
                print(f"[Warn][GUI] unknown option type={rtype}")
                refStr = ""
            return refStr
        else:
            return ""

    def _check_value_validity(self,qEdit):
        """
         Check the validity of the text inserted in QTextEdit widget
         Parameters:
         ---------
         qedit, QTextEdit widget

        """
        flag = False
        if qEdit.__class__== QtWidgets.QLineEdit:
            txt = qEdit.text()
            try:
                val = float(txt)
                if val%0.25==0:
                    flag = True
            except:
                print(f"the value {txt} is invalid ")
                flag = False

        return flag

    def _clearPredictions(self):
        # clear table values
        for i in range(self._patient_info_table.rowCount()):
            for j in range(self._patient_info_table.columnCount()):
                self._patient_info_table.item(i,j).setText("")
        # clear predicted values
        self._lSph_edit.setText("")
        self._lCyl_edit.setText("")
        self._lAx_edit.setText("")
        self._rSph_edit.setText("")
        self._rCyl_edit.setText("")
        self._rAx_edit.setText("")
        self._rAdd_edit.setText("")

        self._rcSph.setText("")
        self._rcCyl.setText("")
        self._rcAx.setText("")
        self._lcSph.setText("")
        self._lcCyl.setText("")
        self._lcAx.setText("")
        self._lAdd_edit.setText("")

    def Predict(self,currentPat):

        prediction = self.vx_predictor.PredictSubjectiveRefraction(currentPat)
        # prediction = self.PredictIOL(prediction)
        ind = currentPat.index[0]
        for eIdx in ['_Right','_Left']:
            sr  = self.surgeryRecommender.Run(currentPat.loc[ind,f'SphericalEquivalent_5{eIdx}'],
                                        currentPat.loc[ind,f'Topo_Sim_K_Avg{eIdx}'],
                                        currentPat.loc[ind,f'Pachy_MEASURE_Thickness{eIdx}'],
                                        currentPat.loc[ind,f'Topo_KERATOCONUS_Kpi{eIdx}'],
                                        prediction.loc[ind,f'Predicted_Cylinder{eIdx}'],
                                        currentPat.loc[ind,f'WF_ZERNIKE_5_HOA{eIdx}'],
                                        currentPat.loc[ind,f'Age'],
                                        currentPat.loc[ind,f'Pachy_MEASURE_Acd{eIdx}'],
                                        23.4,
                                        30,
                                        2,
                                        priorLasik=False,pvd=False,
                                        opticalZoneDiameter=6.5,
                                        dominantEye='right',
                                        flapThickness=100,topoPattern='normal')
                    
        print(sr)
        # add info from the Revo
        # prediction = self._AddRevoData(prediction)

        return prediction

    def _list_select_callback(self,event):
        """Calback for list item selection """
        # clear content of predictions
        self._list_item_change_callback()

        # try:
        # Export the previously predicted values to list and export them
        currentItem     = self._patient_list.currentIndex().row()
        folder_selected = self.file_list_[currentItem]
        #TODO: add  marker for files precviously parsed using list index
        if folder_selected.endswith('zip'):
            pat_data  = self.vx_parser.ParseVXFromZip(os.path.join(self.data_path_,folder_selected))
        else:
            pat_data        = self.vx_parser.ParseFolder(os.path.join(self.data_path_,folder_selected))
        if len(pat_data.index)>1: # if several examinations were parsed
            pat_data = pat_data.loc[pat_data.index[-1],:] # take last one

        self.currentPat = pat_data.copy()
        # self.currentPat = self._AddRevoData(self.currentPat)
        patInds         = self._predictions.index.values==self.currentPat.index.values
        if any(patInds):
            print("[Info][GUI] Found previous measurement")
            self.currentPrediction = self._predictions.loc[patInds,:]
        else:
            self.currentPrediction = self.Predict(self.currentPat)

        # display prediction in panel
        self._display_prediction(self.currentPrediction)
        # except:
        #     self._clearPredictions()

    def _AddRevoData(self,pat_data)->pd.DataFrame:
        """
         Add axial length to the vx120 measurements
        """
        pat_data.loc[pat_data.index,'AxialLength_Left']       = self.iol_predictor.Impute('l_axial_length_mean')
        pat_data.loc[pat_data.index,'AxialLength_Right']      = self.iol_predictor.Impute('r_axial_length_mean')
        pat_data.loc[pat_data.index,'targetRefraction_Left']  = 0
        pat_data.loc[pat_data.index,'targetRefraction_Right'] = 0
        pat_data.loc[pat_data.index,'Aconst_Left']            = 118.9
        pat_data.loc[pat_data.index,'Aconst_Right']           = 118.9

        return pat_data

    def PredictIOL(self, features)->pd.DataFrame:
        """
         Predict IOL power and refraction based on feature values
         Parameters:
         ------------
         Aconst, array(float)
          IOL aconstant
         targetRefraction, array(float)
          target refraction for [left,Right]
         features, DataFrame
          current vx130 measurements
         Output:
         -------
          iolL, iolR, float
           predicted iol power left and right eye
          refL, refR, float
           predicted refraction left and right eye
        """
        # fL = pd.DataFrame(index=[0,1])
        # fKey = 'Age'
        # for eIdx in range(2):
        #     if fKey in features.columns:
        #         fL.loc[eIdx,fKey] = features[fKey].values
        #         if fL[fKey].isna()[eIdx]:
        #             fL.loc[eIdx,fKey] = self.iol_predictor.Impute(fKey,strategy='median')
        #     else:
        #         fL.loc[eIdx,fKey] = self.iol_predictor.Impute(fKey,strategy='median')

        # fKey = 'Topo_Sim_K_Avg'
        # iKey = 'radius_se_mean'
        # eInd = ['_Left','_Right']
        # iInd = ['l_','r_']
        # for eIdx in range(2):
        #     if fKey+eInd[eIdx] in features.columns:
        #         fL.loc[eIdx,fKey] = features[fKey+eInd[eIdx]].values
        #         if fL[fKey].isna()[eIdx]:
        #             meanR =  self.iol_predictor.Impute(iInd[eIdx]+iKey,strategy='median')
        #             meanK = 337.5/meanR
        #             fL.loc[eIdx,fKey] = meanK
        #     else:
        #         meanR =  self.iol_predictor.Impute(iInd[eIdx]+iKey,strategy='median')
        #         meanK = 337.5/meanR
        #         fL.loc[eIdx,fKey] = meanK

        # fKey = 'Pachy_MEASURE_Acd'
        # iKey = 'acd_mean'
        # eInd = ['_Left','_Right']
        # iInd = ['l_','r_']
        # for eIdx in range(2):
        #     if fKey+eInd[eIdx] in features.columns:
        #         fL.loc[eIdx,fKey] = features[fKey+eInd[eIdx]].values
        #         if fL[fKey].isna()[eIdx]:
        #             fL.loc[eIdx,fKey] = self.iol_predictor.Impute(iInd[eIdx]+iKey)
        #     else:
        #         fL.loc[eIdx,fKey] = self.iol_predictor.Impute(iInd[eIdx]+iKey)

        # fKey = 'Pachy_MEASURE_WhiteToWhite'
        # iKey = 'wtw_mean'
        # eInd = ['_Left','_Right']
        # iInd = ['l_','r_']
        # for eIdx in range(2):
        #     if fKey+eInd[eIdx] in features.columns:
        #         fL.loc[eIdx,fKey] = features[fKey+eInd[eIdx]].values
        #         if fL[fKey].isna()[eIdx]:
        #             fL.loc[eIdx,fKey] = self.iol_predictor.Impute(iInd[eIdx]+iKey)
        #     else:
        #         fL.loc[eIdx,fKey] = self.iol_predictor.Impute(iInd[eIdx]+iKey)

        # fKey = 'AxialLength'
        # eInd = ['_Left','_Right']
        # for eIdx in range(2):
        #     if fKey+eInd[eIdx] in features.columns:
        #         fL.loc[eIdx,fKey] = features[fKey+eInd[eIdx]].values
        #         if fL[fKey].isna()[eIdx]:
        #             fL.loc[eIdx,fKey] = self.iol_predictor.Impute(fKey+eInd[eIdx])
        #     else:
        #         fL.loc[eIdx,fKey] = self.iol_predictor.Impute(fKey + eInd[eIdx])

        # fL.loc[0,'targetRefraction'] = features['targetRefraction_Left'].values
        # fL.loc[1,'targetRefraction'] = features['targetRefraction_Right'].values
        # fL.loc[0,'Aconst']           = features['Aconst_Left'].values
        # fL.loc[1,'Aconst']           = features['Aconst_Right'].values
        # predict

        fL = pd.DataFrame(index=[0])
        fL.loc[0,'Age']               = None
        for lIdx in ['_Right','_Left']:
            fL.loc[0,f'Aconst{lIdx}']            = 118.9
            fL.loc[0,f'Pachy_MEASURE_Acd{lIdx}'] = 3.2
            fL.loc[0,f'AxialLength{lIdx}']       = 24
            fL.loc[0,f'Topo_Sim_K_Avg{lIdx}']    = 43.5
            fL.loc[0,f'Pachy_MEASURE_WhiteToWhite{lIdx}'] = 12
            fL.loc[0,f'taergetRefraction{lIdx}'] = 0

        # fR = pd.DataFrame(index=[0])
        # fR.loc[0,'Aconst']            = None
        # fR.loc[0,'Pachy_MEASURE_Acd'] = None
        # fR.loc[0,'AxialLength']       = None
        # fR.loc[0,'Topo_Sim_K_Avg']    = None
        # fR.loc[0,'Pachy_MEASURE_WhiteToWhite'] = None
        # fR.loc[0,'Age'] = None
        # fR.loc[0,'taergetRefraction']= 0
        predictedClass = self.iol_predictor.PredictIolPower(fL,Aconst=118.9,targetRefraction=0)
        # predictedClassR = self.iol_predictor.PredictIolPower(fR,Aconst=118.9,targetRefraction=0)
        # get the IOL power and predicted refraction based on the class
        PL,RL = self.iol_predictor.ComputeAllFormulas(fL['Aconst'][0],
                                                      fL['Topo_Sim_K_Avg'],
                                                      fL['Pachy_MEASURE_Acd'],
                                                      fL['Pachy_MEASURE_WhiteToWhite'],
                                                      fL['AxialLength'],
                                                      fL['targetRefraction'])
        PR,RR = self.iol_predictor.ComputeAllFormulas(fL['Aconst'][1],
                                                      fL['Topo_Sim_K_Avg'],
                                                      fL['Pachy_MEASURE_Acd'],
                                                      fL['Pachy_MEASURE_WhiteToWhite'],
                                                      fL['AxialLength'],
                                                      fL['targetRefraction'])

        iolL = round(4*PL.loc[0,PL.columns[predictedClass[0]]]/4)
        refL = round(4*RL.loc[0,RL.columns[predictedClass[0]]]/4)
        iolR = round(4*PR.loc[0,PR.columns[predictedClass[1]]]/4)
        refR = round(4*RR.loc[0,RR.columns[predictedClass[1]]]/4)

        features.loc[features.index,'Predicted_IOL_Left']                    = iolL
        features.loc[features.index,'Predicted_IOL_Right']                   = iolR
        features.loc[features.index,'Predicted_IOL_Postop_Refraction_Left']  = refL
        features.loc[features.index,'Predicted_IOL_Postop_Refraction_Right'] = refR

        return features

    def _display_prediction(self,patInfo):
        """
            display current patient information in the gui
            Parameters:
            -----------
            prediction - DataFrame,
                the data frame structure output from
        """
        # deg    = u"\N{DEGREE SIGN}"
        self._patient_info_table.itemChanged.disconnect()
        sphRight       = f"{patInfo['Predicted_Sphere_Right'][0]:.2f}"
        cylRight       = f"{patInfo['Predicted_Cylinder_Right'][0]:.2f}"
        axRight        = f"{patInfo['Predicted_Axis_Right'][0]:.0f}"
        sphLeft        = f"{patInfo['Predicted_Sphere_Left'][0]:.2f}"
        cylLeft        = f"{patInfo['Predicted_Cylinder_Left'][0]:.2f}"
        axLeft         = f"{patInfo['Predicted_Axis_Left'][0]:.0f}"
        additionLeft   = f"{patInfo['Predicted_Add_Left'][0]:.2f}"
        additionRight  = f"{patInfo['Predicted_Add_Right'][0]:.2f}"
        cSphRight      = f"{patInfo['Predicted_Contact_Sphere_Right'][0]:.2f}"
        cCylRight      = f"{patInfo['Predicted_Contact_Cylinder_Right'][0]:.2f}"
        cAxRight       = f"{patInfo['Predicted_Contact_Axis_Right'][0]:.0f}"
        cSphLeft       = f"{patInfo['Predicted_Contact_Sphere_Left'][0]:.2f}"
        cCylLeft       = f"{patInfo['Predicted_Contact_Cylinder_Left'][0]:.2f}"
        cAxLeft        = f"{patInfo['Predicted_Contact_Axis_Left'][0]:.0f}"
        # iolLeft        = f"{patInfo['Predicted_IOL_Left'][0]:.0f}"
        # iolRight       = f"{patInfo['Predicted_IOL_Right'][0]:.0f}"
        # iolRefLeft     = f"{patInfo['Predicted_IOL_Postop_Refraction_Left'][0]:.0f}"
        # iolRefRight    = f"{patInfo['Predicted_IOL_Postop_Refraction_Right'][0]:.0f}"

        # populate the patient info table
        self._patient_info.item(0,0).setText(f"{patInfo['Surname'][0]}")
        self._patient_info.item(1,0).setText(f"{patInfo['Firstname'][0]}")
        self._patient_info.item(2,0).setText(f"{patInfo['Sex'][0]}")
        self._patient_info.item(3,0).setText(f"{patInfo['Age'][0]:.2f}")

        self._patient_info_table.item(0,0).setText(f"{patInfo['WF_SPHERE_R_3_Right'][0]:.2f}")
        self._patient_info_table.item(0,1).setText(f"{patInfo['WF_SPHERE_R_3_Left'][0]:.2f}")
        self._patient_info_table.item(1,0).setText(f"{patInfo['WF_CYLINDER_R_3_Right'][0]:.2f}")
        self._patient_info_table.item(1,1).setText(f"{patInfo['WF_CYLINDER_R_3_Left'][0]:.2f}")
        self._patient_info_table.item(2,0).setText(f"{patInfo['WF_AXIS_R_3_Right'][0]:.0f}")
        self._patient_info_table.item(2,1).setText(f"{patInfo['WF_AXIS_R_3_Left'][0]:.0f}")
        self._patient_info_table.item(3,0).setText(f"{patInfo['SphericalEquivalent_3_Right'][0]:.3f}")
        self._patient_info_table.item(3,1).setText(f"{patInfo['SphericalEquivalent_3_Left'][0]:.3f}")
        self._patient_info_table.item(4,0).setText(f"{patInfo['Pachy_MEASURE_Acd_Right'][0]:.2f}")
        self._patient_info_table.item(4,1).setText(f"{patInfo['Pachy_MEASURE_Acd_Left'][0]:.2f}")
        self._patient_info_table.item(5,0).setText(f"{patInfo['Pachy_MEASURE_WhiteToWhite_Right'][0]:.2f}")
        self._patient_info_table.item(5,1).setText(f"{patInfo['Pachy_MEASURE_WhiteToWhite_Left'][0]:.2f}")
        self._patient_info_table.item(6,0).setText(f"{patInfo['Topo_Sim_K_K1_Right'][0]:.2f}")
        self._patient_info_table.item(6,1).setText(f"{patInfo['Topo_Sim_K_K1_Left'][0]:.2f}")
        self._patient_info_table.item(7,0).setText(f"{patInfo['Topo_Sim_K_K2_Right'][0]:.2f}")
        self._patient_info_table.item(7,1).setText(f"{patInfo['Topo_Sim_K_K2_Left'][0]:.2f}")
        self._patient_info_table.item(8,0).setText(f"{patInfo['Pachy_MEASURE_Thickness_Right'][0]:.2f}")
        self._patient_info_table.item(8,1).setText(f"{patInfo['Pachy_MEASURE_Thickness_Left'][0]:.2f}")
        # self._patient_info_table.item(9,0).setText(f"{patInfo['AxialLength_Right'][0]:.2f}")
        # self._patient_info_table.item(9,1).setText(f"{patInfo['AxialLength_Left'][0]:.2f}")
        # self._patient_info_table.item(10,0).setText(f"{patInfo['targetRefraction_Right'][0]:.2f}")
        # self._patient_info_table.item(10,1).setText(f"{patInfo['targetRefraction_Left'][0]:.2f}")
        # self._patient_info_table.item(11,0).setText(f"{patInfo['Aconst_Right'][0]:.2f}")
        # self._patient_info_table.item(11,1).setText(f"{patInfo['Aconst_Left'][0]:.2f}")


        # update edit boxes
        self._lSph_edit.setText(sphLeft)
        self._lCyl_edit.setText(cylLeft)
        self._lAx_edit.setText(axLeft)
        self._rSph_edit.setText(sphRight)
        self._rCyl_edit.setText(cylRight)
        self._rAx_edit.setText(axRight)
        self._rAdd_edit.setText(additionRight)

        self._rcSph.setText(cSphRight)
        self._rcCyl.setText(cCylRight)
        self._rcAx.setText(cAxRight)
        self._lcSph.setText(cSphLeft)
        self._lcCyl.setText(cCylLeft)
        self._lcAx.setText(cAxLeft)
        self._lAdd_edit.setText(additionLeft)

        # self._rIOL.setText(iolRight)
        # self._lIOL.setText(iolLeft)
        # self._rIolRef.setText(iolRefRight)
        # self._lIolRef.setText(iolRefLeft)

        self._patient_info_table.itemChanged.connect(self._patient_table_edit_callback)

    def _update_File_list(self,event):
        """ Update the list of files in the list """
        folderName = os.path.split(event.src_path)[1]

        # update file list
        self.file_list_.insert(0,folderName)
        self._patient_list.insertItem(0,folderName.replace("#"," ").replace('export_patient_','').replace('_',' ').replace('.zip',''))
        self._patient_list.update()

    def _ConstructFolderMonitor(self):
        #TODO: add delete item callback
        if self.monitorVxFolder:
            print("[GUI][Info] Starting folder monitor")
            self._vxEventHandler = PatternMatchingEventHandler(patterns="*")
            self._vxEventHandler.on_created = self._update_File_list
            self._vxObserver     = PollingObserver()
            # Make sure vx path is available. Otherwise, track a dummy folder until vx path is accessible
            if os.path.exists(self.data_path_):
                self._vxObserver.schedule(self._vxEventHandler, self.data_path_,recursive=False)
            else:
                # create a dummy folder inside main directory and start tracking it
                tempFolder = os.path.join(os.getcwd(),'.temp')
                if not os.path.exists(tempFolder):
                    os.mkdir(tempFolder)
                self._vxObserver.schedule(self._vxEventHandler,tempFolder,recursive=False)
            self._vxObserver.start()

    def _setxmlValue(self,root,nPath,val):
        if val.__class__==str:
            flag = True
            k = root
            for n in nPath:
                try:
                    k = k.find(n)
                except:
                    flag = False
                    break
            if flag:
                k.text = str(val)
        return root

    def ExportPredictionToVx40xml(self,prediction,folderName):
        """
            Generate a vx40 xml from predicted values and export it
            to a designated folder, for storage or reading by EMR
            Parameters:
            -----------
            prediction, DataFrame
                a dataframe with columns: ID, FirstName, surname, BirthDate, Sphere_Right/_Left, Cylinder_right/_Left, Axis_Right/_Left
            folderName, str
             a path to the folder in which to place the xml
             the name of the xml file has a fixed pattern and
             will be save as: folderName/patientID_predicted.xml"
        """

        # insert under <subbjective measurements> <LSM_measurements> <measure> <sphere> <cylinder> <axis>
        # to be able to pass to the Oplus
        self._vx40_xml = ET.parse(self._vx40_xml_path_)
        root           = self._vx40_xml.getroot()
        # pKeys   = prediction.keys()
        n      = dateutil.parser.parse('010100').now()
        # lsm measurement
        self._setxmlValue(root,['optometry','company'],"Mikajaki")
        self._setxmlValue(root,['optometry','model_name'],"AutoRef")
        self._setxmlValue(root,['optometry','date'],f"{n.day:02d}/{n.month:02d}/{n.year}")
        self._setxmlValue(root,['optometry','time'],f"{n.hour:02d}:{n.minute:02d}")
        self._setxmlValue(root,['optometry','patient','ID'],str(prediction['ID'][0]))
        self._setxmlValue(root,['optometry','patient','first_name'],str(prediction['Firstname'][0]))
        self._setxmlValue(root,['optometry','patient','last_name'],str(prediction['Surname'][0]))
        self._setxmlValue(root,['optometry','patient','gender'],"m" if prediction['Gender'][0]==0.0 else "f")
        self._setxmlValue(root,['optometry','patient','birthday'],str(prediction['BirthDate'][0]))

        self._setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_right','sphere'],f"{prediction['Predicted_Sphere_Right'][0]}")
        self._setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_right','cylinder'],f"{prediction['Predicted_Cylinder_Right'][0]}")
        self._setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_right','axis'],f"{prediction['Predicted_Axis_Right'][0]}")
        self._setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_right','addition'],f"{prediction['Predicted_Add_Right'][0]}")

        self._setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_left','sphere'],f"{prediction['Predicted_Sphere_Left'][0]}")
        self._setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_left','cylinder'],f"{prediction['Predicted_Cylinder_Left'][0]}")
        self._setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_left','axis'],f"{prediction['Predicted_Axis_Left'][0]}")
        self._setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_left','addition'],f"{prediction['Predicted_Add_Left'][0]}")

        # contact lenses
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','sphere'],f"{prediction['Predicted_Contact_Sphere_Right'][0]}")
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','cylinder'],f"{prediction['Predicted_Contact_Cylinder_Right'][0]}")
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','axis'],f"{prediction['Predicted_Contact_Axis_Right'][0]}")
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','manufacturer'],'Johnson & Johnson')
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','model'],'1-Day Acuvue Moist 180')
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','diameter'],'14.2')
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','base_curve'],'8.5')


        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','sphere'],f"{prediction['Predicted_Contact_Sphere_Left'][0]}")
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','cylinder'],f"{prediction['Predicted_Contact_Cylinder_Left'][0]}")
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','axis'],f"{prediction['Predicted_Contact_Axis_Left'][0]}")
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','model'],'1-Day Acuvue Moist 180')
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','diameter'],'14.2')
        self._setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','base_curve'],'8.5')

        xmlName = str(prediction['ID'][0]).replace('/','_')+"_predicted.xml"
        self._vx40_xml.write(os.path.join(folderName,xmlName))
        print(f"[GUI][Info] Exporting patient xml: {xmlName}")
