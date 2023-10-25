#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Thu Dec 26 13:16:19 2019

    @author: Ofir Shukron
    @description: This file containes tools to parse vx130/120 measurements from zip, folder, and xml file
"""

import os
import re
from shutil import rmtree
import pandas as pd
from zipfile import ZipFile
import xml.etree.ElementTree as ET

# TODO: add logger
class Parser:

    def __init__(self,vertexDistance=0.012):
        self.vertexDistance  = vertexDistance # vertex distance in m TODO: move to parameters
        self._vxUnpackFolder = os.path.join(os.path.dirname(__file__),'.tmp')

        if not os.path.isdir(self._vxUnpackFolder):
            os.mkdir(self._vxUnpackFolder)

        print("[Info][VXparser] vx Parser class initialized")

    def ParseVXFromZip(self,zipPath):
        """
         Unpack a vx zip files into client db folder and parse vx measurements
         append the data into the current vx db and save the new database

         Parameters:
         ------------
         zipPath, str
             the path to the zip file containing vx measurements
         imputeStrategy, optional, str, default="median"
            the strategy to impute missig values. currently implemented
            "median" and "mean" strategies
         Output:
         -------
         newData, DataFrame
            parsed patient measurements
        """
        if zipPath is None:
            return pd.DataFrame(index=[0])
        else:
            if not os.path.exists(zipPath):
                raise ValueError(f"Cannot find the file {zipPath}")

            newData = pd.DataFrame()
            _, fileExt = os.path.splitext(zipPath)
            if fileExt=='.zip':
                print(f"[INFO][VXparser] Extracting file {zipPath}")
                try:
                    with ZipFile(zipPath, 'r') as zip_ref:
                        zip_ref.extractall(self._vxUnpackFolder)
                    # Parse the client db and add to the internal db
                    # try parsing the zip file
                    patFolder = os.listdir(os.path.join(self._vxUnpackFolder,'clientdb'))

                    newData,_ = self.ParseFolder(os.path.join(self._vxUnpackFolder,'clientdb',patFolder[0]))

                    # Delete extracted files
                    l = os.listdir(self._vxUnpackFolder)
                    for lIdx in l:
                        rmtree(os.path.join(self._vxUnpackFolder,lIdx))
                except:
                    print(f"[Error][VXparser] Cannot parse {zipPath}")

            return newData

    def BatchParse(self,folderPath):
        """
          Sequentially parse all vx examinations in a folder.
          The batch parsing uses the ParseFolder method
          Parameters:
          ------------
          folderPath, str
           a path to the folder in which vx120 examination are located.
           examination are either zip files or extracted zip, containing folders
          Output:
          data, DataFrame
           a dataFrame with number of rows matching number of valid vx folders
           each row contains the values parsed from the folder heirarchy
           missing values are set to Nan
        """
        if not os.path.exists(folderPath):
            raise ValueError(f"Cannot find the folder {folderPath}")

        fl   = os.listdir(folderPath)
        data = pd.DataFrame()
        next = -1
        for fIdx in fl:
            if os.path.isdir(os.path.join(folderPath,fIdx)):
                print(f'[Info][VXparser] Parsing {next}/{len(fl)} folder:{os.path.join(folderPath,fIdx)}')
                examData,_ = self.ParseFolder(os.path.join(folderPath,fIdx))
                for eIdx in examData.index:
                    next+=1
                    for cIdx in examData.columns:
                        data.loc[next,cIdx] = examData.loc[eIdx,cIdx]
            elif fIdx.endswith('zip'):
                print(f'[Info][VXparser] Parsing {next}/{len(fl)} zipfile:{os.path.join(folderPath,fIdx)}')
                examData = self.ParseVXFromZip(os.path.join(folderPath,fIdx))
                for eIdx in examData.index:
                    next+=1
                    for cIdx in examData.columns:
                        data.loc[next,cIdx] = examData.loc[eIdx,cIdx]

        return data

    def ParseResultsFile(self,resultFilePath):
        """
            Parse the vx130 Results.txt file
            Output:
            -------
            res, DataFrame
             a DataFrame containing the fields of the txt file and their values
        """
        if not os.path.exists(resultFilePath):
            raise ValueError(f"Cannot find the file {resultFilePath}")
        res = pd.DataFrame() # results for current examination
        os.path.isfile(resultFilePath)
        dLines = self.ReadLines(resultFilePath)
        for l in dLines:
            # find header name inside brackets
            headerName = re.findall(r'(?<=\[)\w+(?=\])',l)
            if len(headerName)>0:
                row = False # indicator for a row or header
                currentHeader = headerName[0]
                if currentHeader.find('_L_1')!=-1:
                    eyeInd = '_Left'
                    currentHeader = currentHeader.replace('_L_1','')
                elif currentHeader.find('_R_1')!=-1:
                    eyeInd = '_Right'
                    currentHeader = currentHeader.replace('_R_1','')
                else:
                    eyeInd = '' # no
            else:
                row = True
            # Parse rows
            if row:
                val = l.split("=")
                if len(val)>1:
                    fieldName = val[0]
                    fieldVal  = val[1].replace('\n','')
                else:
                    fieldName = val
                    fieldVal  = pd.NA

                f = currentHeader+'_'+fieldName+eyeInd
                res.loc[0,f] = fieldVal
        return res

    def ParseFolder(self,patFolder):
        if not os.path.exists(patFolder):
            raise ValueError(f"Cannot find the folder {patFolder}")
        examFolders = os.listdir(patFolder)
        currentRow  = pd.DataFrame()
        ind         = 0
        examNames   = ['Retro','Tono','Topo','WF','ARK']
        credentials = ['Firstname','Surname','CurrentDate','CurrentTime','BirthDate','Sex']
        measFlag    = pd.Series() # indicator for the presence of measurements

        for eIdx in examFolders:
            if os.path.isdir(os.path.join(patFolder,eIdx)):
                for exmIdx in examNames:
                    for lIdx in ['_Right','_Left']:
                        measFlag[exmIdx+lIdx]  = False

                    if os.path.isdir(os.path.join(patFolder,eIdx,exmIdx)):
                        tFiles   = os.listdir(os.path.join(patFolder,eIdx,exmIdx))
                        header   = exmIdx
                        for tIdx in tFiles:
                            if tIdx.endswith('.txt'):
                                fieldName,indicator = self._GetMeasurementIndicator(exmIdx,tIdx)
                                measFlag[fieldName] = indicator

                                txtVals = self._ParseTxtFile(os.path.join(patFolder,eIdx,header,tIdx))
                                for cIdx in txtVals.columns:
                                    if cIdx in credentials:
                                        currentRow.loc[ind,cIdx] = txtVals.loc[0,cIdx]
                                    else:
                                        currentRow.loc[ind,header+'_'+cIdx] = txtVals.loc[0,cIdx]
                            elif tIdx.endswith('.raw'):
                                # parse images
                                pass

                if os.path.isdir(os.path.join(patFolder,eIdx,'Pachy')):
                    header = 'Pachy'
                    for lIdx in ['_Right','_Left']:
                        measFlag['Pachy'+lIdx]     = False
                        measFlag['MultiSlit'+lIdx] = False

                    pFiles = os.listdir(os.path.join(patFolder,eIdx,header))
                    for pIdx in pFiles:
                        if pIdx.endswith('.txt'):
                            fieldName,indicator = self._GetMeasurementIndicator('Pachy',pIdx)
                            measFlag[fieldName] = indicator

                            txtVals = self._ParseTxtFile(os.path.join(patFolder,eIdx,header,pIdx))
                            for cIdx in txtVals.columns:
                                if cIdx in credentials:
                                    currentRow.loc[ind,cIdx] = txtVals.loc[0,cIdx]
                                else:
                                    currentRow.loc[ind,header+'_'+cIdx] = txtVals.loc[0,cIdx]
                        elif os.path.isdir(os.path.join(patFolder,eIdx,header,pIdx)):
                            header2 = 'MultiSlit'
                            mFiles  = os.listdir(os.path.join(patFolder,eIdx,header,pIdx))
                            for mIdx in mFiles:
                                if mIdx.endswith('.txt'):
                                    fieldName,indicator = self._GetMeasurementIndicator('Pachy_MultiSlit',pIdx)
                                    measFlag[fieldName] = indicator
                                    txtVals = self._ParseTxtFile(os.path.join(patFolder,eIdx,header,pIdx,mIdx))
                                    for cIdx in txtVals.columns:
                                        if cIdx in credentials:
                                            currentRow.loc[ind,cIdx] = txtVals.loc[0,cIdx]
                                        else:
                                            currentRow.loc[ind,header+'_'+header2+'_'+cIdx] = txtVals.loc[0,cIdx]
                                elif mIdx.endswith('.raw'):
                                    # parse images
                                    pass
                ind+=1
        # a special care for the Pachy directory
        return currentRow, measFlag

    def _ParseTxtFile(self,filePath):
        """
          Extract data from a results.txt file
        """
        if not os.path.exists(filePath):
            raise ValueError(f"Cannot find the file {filePath}")

        # currentRow = None
        # if filePath.endswith('.txt'): # only parse txt files
        currentRow = pd.DataFrame()
        # check if file is for the right or left eye

        eyeInd   = re.findall(r'left|Left|right|Right',os.path.basename(filePath))
        if len(eyeInd)==1:
            eyeInd   = eyeInd[0]
            eyeInd   = eyeInd[0].upper()+eyeInd[1:]
        else:
            eyeInd = pd.NA
        # read all lines
        lines = self.ReadLines(filePath)
        try:
            if len(lines)>0:
                for lIdx in lines:
                    val = lIdx.split("=")
                    if len(val)==1:
                        currentHeader = val[0].replace('\n','').replace('[','').replace(']','')
                    elif len(val)==2: # field name = value
                        if currentHeader =='PATIENT':
                            if val[0] in ['Surname','Firstname','CurrentDate','CurrentTime','BirthDate','Sex']:
                                currentRow.loc[0,val[0]] = val[1].replace('\n','')
                            else:
                                currentRow.loc[0,val[0]+'_'+eyeInd] = float(val[1].replace('\n',''))
                        else: # measurement
                            fieldName  = currentHeader + '_' + val[0] + '_' + eyeInd
                            currentVal = val[1]
                            # parse only scalars and ignore vectors
                            if len(currentVal.split(' '))==1:
                                currentVal = currentVal.replace('\n','') # remove eol symbol and translate to float
                                if currentVal.find('#')==-1: # avoid cases for which #QNAN0 appears
                                    currentVal = pd.NA if pd.isnull(currentVal) else float(currentVal)
                                    # assign value in field
                                    currentRow.loc[0, fieldName] = currentVal

        except:
            print(f"[Error][VXparser] Problem loading file {filePath} fieldName: {fieldName} value {currentVal}")

        return currentRow

    def ReadLines(self,filePath):
        """
             Read a txt file and return all lines
        """
        isUtf16  = False
        # isUtf8   = False
        # fileOpen = False
        lines    =  []
        try: # try first the utf-16
            with open(filePath,'r', encoding='utf-16') as file:
                lines = file.readlines()
            # fileOpen = True
            isUtf16  = True
        except:
            # fileOpen = False
            isUtf16  = False

        if isUtf16==False: # else try utf-8
            try:
                with open(filePath,'r',encoding='utf-8') as file:
                    lines = file.readlines()
                # fileOpen = True
                # isUtf8   = True
            except:
                fileOpen = False
                # isUtf8   = False
        return lines

    @staticmethod
    def _GetMeasurementIndicator(exmName,fileName):
        if re.findall('left',fileName,re.IGNORECASE):
            return exmName+'_Left', True
        elif re.findall('right',fileName,re.IGNORECASE):
            return exmName+'_Right', True
