#Prevot EMR Parser

#import os
import pandas as pd
import numpy as np
import ParserUtils
import re
import subprocess
import os
import dateutil
import datetime


class Parser():
    """
      Parsing utilities to extract information from the Studio visio EMR system of Dr. E. Prevot
      The class contains methods to translate the mdb files to csv and then to parse the results

      To run a full pipeline, use the Pipeline method with an mdb file

      Otherwise, for a parsed mdb with consultation, patients and refraction csv files
      use the Load method and the the Parse method

    """

    def __init__(self):
     print('[Parser][Info] Class initialized')
     self._data_loaded = False  # indicator for loaded datasets

    def Load(self,patientFile,refractionFile,consultationFile):
        '''
            Parameters
            ------------
            patientFile: str
                 a path to the Patients_Prevot.csv file
            refractionFile: str,
                a path to the tREFRACTION_Prevot.csv file
            consultationFile: str
                a path to Consultation_Prevot.csv file
        '''
        pat  = pd.read_csv(patientFile,low_memory=False)
        ref  = pd.read_csv(refractionFile, low_memory=False)
        cons = pd.read_csv(consultationFile, low_memory=False)

        nameMap = {'NOM':'Surname',
                   'Prénom':'FirstName',
                   'Date de Naissance':'BirthDate',
                   'Date naissance':'BirthDate',
                   'N° consultation':'NumConsultation',
                   'NumConsult':'NumConsultation',
                   'REFRACTION':'Refraction',
                   'REMARQUES':'Comments',
                   'LUNETTES':'Glasses',
                   'LENTILLES':'ContactLenses',
                   'Code patient':'PatientCode',
                   'Code Patient':'PatientCode',
                   'Code acte':'ActeCode',
                   'SEXE':'Gender',
                   'Adresse':'Address',
                   'SphD':'AutoRefractometerSphere_Right',
                   'SphG':'AutoRefractometerSphere_Left',
                   'CylD':'AutoRefractometerCylinder_Right',
                   'CylG':'AutoRefractometerCylinder_Left',
                   'AxeD':'AutoRefractometerAxis_Right',
                   'AxeG':'AutoRefractometerAxis_Left',
                   'AddG':'AutoRefractometerAdd_Left',
                   'AddD':'AutoRefractometerAdd_Right',
                   'TOG':'Tonometry_Left',
                   'TOD':'Tonometry_Right'}

        # rename columns in data tables
        pat.rename(columns=nameMap, inplace=True)
        ref.rename(columns=nameMap,inplace=True)
        cons.rename(columns=nameMap,inplace=True)

        self.patients     = pat
        self.refraction   = ref
        self.consultation = cons
        self._data_loaded = True

    def _DataStats(self):

        def _lenNZ(l):
            """function to measure the length of list entry in a DataFrame """
            if isinstance(l,list):
                return len(l)
            else:
                return 0

        if self._data_loaded:
            numPat     = len(self.patients['PatientCode'].unique())
            numSubjRef = (self.consultation['Refraction'].str.findall('subjective',flags=re.IGNORECASE).apply(_lenNZ)>=1).sum()
            numCons    = len(self.consultation['NumConsultation'].dropna().unique())
            p_code     = self.consultation['PatientCode'].unique()
            p_hist     = np.zeros(p_code.max()+1)
            # construct histogram of patients
            for pIdx in self.consultation['PatientCode']:
                p_hist[pIdx]+=1

            print('___Data Summary____')
            print(f'Num. unique patients: {numPat}')
            print(f'Num. Consultations: {numCons}')
            print(f'Num. Subj. refractions {numSubjRef}')
        else:
            print('Please load data first')

    def Parse(self):
        '''
            Parse a database after loading (see Load() method). Then, merge them into one unified DB

            For each consultation in the Consultation db, match the patient accroding
            to the patient code found in both Consultation and Patients db and pool the data
            together

        '''
        if not self._data_loaded:
            print('[Info][Parser] Please load the csv datasets Patients, Refraction, Consultation (see the Load() method for details) before running Parse()')
            return 1

        # translate French special characters to English
        # f2eDict      = ParserUtils.Utils.FrenchToEnglishDateStr()
        consultation = self.consultation
        lc           = len(consultation)

        # preallocations
        self.data   = pd.DataFrame() # re-initialize the structure
        patCode     = np.ndarray(shape=lc,dtype=int)
        numConsult  = np.ndarray(shape=lc,dtype=int)
        firstName   = list()
        surname     = list()
        birthDate   = list()
        gender      = list()
        examDate    = list()
        age         = list()
        tonometry   = np.nan*np.zeros(shape = (lc,2),dtype=np.float)
        pachimetry  = np.nan*np.zeros(shape = (lc,2),dtype=np.float)
        keratometry = np.nan*np.zeros(shape = (lc,4),dtype=np.float)
        objective   = np.nan*np.zeros(shape = (lc,6),dtype=np.float)
        subjective  = np.nan*np.zeros(shape = (lc,6),dtype=np.float)
        glasses     = np.nan*np.zeros(shape = (lc,6),dtype=np.float)
        refraction  = np.nan*np.zeros(shape = (lc,8),dtype=np.float)

        for cIdx in range(lc):
            print(f'Parsing  {cIdx+1}/{lc}')
            patCode[cIdx]    = np.int(consultation.loc[consultation.index[cIdx], 'PatientCode'])
            # numConsult[cIdx] = np.int(cIdx)
            numConsult[cIdx] = int(consultation.loc[consultation.index[cIdx],"NumConsultation"])
            pIdx             = consultation.loc[consultation.index[cIdx],'PatientCode']
            patInds          = np.where(self.patients['PatientCode']==pIdx)[0]
            # for each consultation find the matching patient
            if len(patInds)>0:
                patInd       = patInds[0]
                firstName.append(ParserUtils.Utils.ToLowerCase(self.patients.loc[self.patients.index[patInd],'FirstName']))
                surname.append(ParserUtils.Utils.ToLowerCase(self.patients.loc[self.patients.index[patInd],'Surname']))
                gender.append(self.patients.loc[self.patients.index[patInd],'Gender'])
                ref = self.refraction.loc[self.refraction['NumConsultation']==numConsult[cIdx]]

                if len(ref)>=1:
                    # print(f'[Parser] Found entry of refraction for consultation {numConsult[cIdx]}')
                    ref_vals = np.zeros((len(ref),8))
                    for vIdx in range(len(ref)):
                        ref_vals[vIdx,0] = self._ParseRefValFromStr(ref.loc[ref.index[vIdx],'AutoRefractometerSphere_Right'])
                        ref_vals[vIdx,1] = self._ParseRefValFromStr(ref.loc[ref.index[vIdx],'AutoRefractometerCylinder_Right'])
                        ref_vals[vIdx,2] = self._ParseRefValFromStr(ref.loc[ref.index[vIdx],'AutoRefractometerAxis_Right'])
                        ref_vals[vIdx,3] = self._ParseRefValFromStr(ref.loc[ref.index[vIdx],'AutoRefractometerAdd_Right'])
                        ref_vals[vIdx,4] = self._ParseRefValFromStr(ref.loc[ref.index[vIdx],'AutoRefractometerSphere_Left'])
                        ref_vals[vIdx,5] = self._ParseRefValFromStr(ref.loc[ref.index[vIdx],'AutoRefractometerCylinder_Left'])
                        ref_vals[vIdx,6] = self._ParseRefValFromStr(ref.loc[ref.index[vIdx],'AutoRefractometerAxis_Left'])
                        ref_vals[vIdx,7] = self._ParseRefValFromStr(ref.loc[ref.index[vIdx],'AutoRefractometerAdd_Left'])
                    # take the non NaN values
                    v = np.zeros(8)*np.nan
                    for rvIdx in range(8):
                        v =  ref_vals[np.isnan(ref_vals[:,rvIdx])==False,rvIdx]
                        refraction[cIdx,rvIdx] = v[0] if len(v)>0 else np.nan

                    # correct for sphere non zero but cyl and axis = np.nan

                    refraction[cIdx,1] = 0 if (np.isnan(refraction[cIdx,0])==False)&(np.isnan(refraction[cIdx,1])) else refraction[cIdx,1]
                    refraction[cIdx,2] = 0 if (np.isnan(refraction[cIdx,0])==False)&(np.isnan(refraction[cIdx,2])) else refraction[cIdx,2]

                    refraction[cIdx,5] = 0 if (np.isnan(refraction[cIdx,4])==False)&(np.isnan(refraction[cIdx,5])) else refraction[cIdx,5]
                    refraction[cIdx,6] = 0 if (np.isnan(refraction[cIdx,4])==False)&(np.isnan(refraction[cIdx,6])) else refraction[cIdx,6]
                else:
                    refraction[cIdx,:] = np.nan
                # if np.isnan(refraction[cIdx,0])&(np.isnan(refraction[cIdx,1])==False):
                #     print('something is wrong here')

                # parse exam date, birth date and age
                bDate = self.patients.loc[self.patients.index[patInd],'BirthDate']
                eDate = consultation.loc[consultation.index[cIdx],'Date']
                bdate_str,edate_str,a = self._ResolveAgeByDates(bDate,eDate,dayfirst=False,yearfirst=False)
                birthDate.append(bdate_str)
                examDate.append(edate_str)
                age.append(a)
            else:
                print(f'Patient {pIdx} was not found')
                firstName.append(np.nan)
                surname.append(np.nan)
                birthDate.append(np.nan)
                gender.append(np.nan)
                examDate.append(np.nan)
                age.append(np.nan)

            tonometry[cIdx,0] = consultation.loc[consultation.index[cIdx],'Tonometry_Left']
            tonometry[cIdx,1] = consultation.loc[consultation.index[cIdx],'Tonometry_Right']

            refStr = consultation.loc[consultation.index[cIdx],'Refraction']
            if isinstance(refStr,str):
                s = refStr.replace(' ','').split('\n\n')# split into blocks
                for sIdx in range(len(s)): # for each block
                    if len(re.findall('lunettes\s?port[ée]es',s[sIdx],flags=re.IGNORECASE))>=1:
                        glasses[cIdx,0],glasses[cIdx,1],glasses[cIdx,2],glasses[cIdx,3],glasses[cIdx,4],glasses[cIdx,5] = self._ParseRefraction(s[sIdx])
                    elif len(re.findall('auto\s?r[ée]fractom[èe]tre',s[sIdx],flags=re.IGNORECASE))>=1:
                        objective[cIdx,0],objective[cIdx,1],objective[cIdx,2],objective[cIdx,3],objective[cIdx,4],objective[cIdx,5] = self._ParseRefraction(s[sIdx])
                    elif len(re.findall('r[ée]fraction\s?subjective',s[sIdx],flags=re.IGNORECASE))>=1:
                        subjective[cIdx,0],subjective[cIdx,1],subjective[cIdx,2],subjective[cIdx,3],subjective[cIdx,4],subjective[cIdx,5] = self._ParseRefraction(s[sIdx])
                    elif len(re.findall('k[ée]ratom[ée]trie',s[sIdx],flags=re.IGNORECASE))>=1:
                        # serach for missing line below
                        if (sIdx+2)<=len(s):
                            if len(re.findall('rog',s[sIdx+1],flags=re.IGNORECASE))>=1:
                                s[sIdx] = s[sIdx]+'\n'+s[sIdx+1]
                        keratometry[cIdx,0], keratometry[cIdx,1],keratometry[cIdx,2],keratometry[cIdx,3]= self._ParseKeratometry(s[sIdx])
                    elif len(re.findall('pachymétrie',s[sIdx],flags=re.IGNORECASE))>=1:
                        sInd = s[sIdx].lower().index('pachymétrie')
                        pachimetry[cIdx,0], pachimetry[cIdx,1] = self._ParsePachimetry(s[sIdx][sInd:])

        self.data['PatientCode']                     = patCode
        self.data['NumConsult']                      = numConsult
        self.data['FirstName']                       = firstName
        self.data['Surname']                         = surname
        self.data['BirthDate']                       = birthDate
        self.data['ExamDate']                        = examDate
        self.data['Age']                             = age
        self.data['Gender']                          = gender
        self.data['AutoRefractometerSphere_Left']    = objective[:,0]
        self.data['AutoRefractometerSphere_Right']   = objective[:,1]
        self.data['AutoRefractometerCylinder_Left']  = objective[:,2]
        self.data['AutoRefractometerCylinder_Right'] = objective[:,3]
        self.data['AutoRefractometerAxis_Left']      = objective[:,4]
        self.data['AutoRefractometerAxis_Right']     = objective[:,5]
        # self.data['VXSphere_Left']                   = objective[:,0]
        # self.data['VXSphere_Right']                  = objective[:,1]
        # self.data['VXCylinder_Left']                 = objective[:,2]
        # self.data['VXCylinder_Right']                = objective[:,3]
        # self.data['VXAxis_Left']                     = objective[:,4]
        # self.data['VXAxis_Right']                    = objective[:,5]
        self.data['VisualAcuitySphere_Left']         = subjective[:,0]
        self.data['VisualAcuitySphere_Right']        = subjective[:,1]
        self.data['VisualAcuityCylinder_Left']       = subjective[:,2]
        self.data['VisualAcuityCylinder_Right']      = subjective[:,3]
        self.data['VisualAcuityAxis_Left']           = subjective[:,4]
        self.data['VisualAcuityAxis_Right']          = subjective[:,5]
        self.data['GlassesSphere_Left']              = glasses[:,0]
        self.data['GlassesSphere_Right']             = glasses[:,1]
        self.data['GlassesCylinder_Left']            = glasses[:,2]
        self.data['GlassesCylinder_Right']           = glasses[:,3]
        self.data['GlassesAxis_Left']                = glasses[:,4]
        self.data['GlassesAxis_Right']               = glasses[:,5]
        self.data['Tonometry_Left']                  = tonometry[:,0]
        self.data['Tonometry_Right']                 = tonometry[:,1]
        self.data['Pachimetry_Left']                 = pachimetry[:,0]
        self.data['Pachimetry_Right']                = pachimetry[:,1]
        self.data['K1_Left']                         = keratometry[:,0]
        self.data['K1_Right']                        = keratometry[:,1]
        self.data['K2_Left']                         = keratometry[:,2]
        self.data['K2_Right']                        = keratometry[:,3]

        self.data['Ref_Sphere_Right']                = refraction[:,0]
        self.data['Ref_Cylinder_Right']              = refraction[:,1]
        self.data['Ref_Axis_Right']                  = refraction[:,2]
        self.data['Ref_Add_Right']                   = refraction[:,3]
        self.data['Ref_Sphere_Left']                 = refraction[:,4]
        self.data['Ref_Cylinder_Left']               = refraction[:,5]
        self.data['Ref_Axis_Left']                   = refraction[:,6]
        self.data['Ref_Add_Left']                    = refraction[:,7]

        return self.data

    def _ExtractSphCylAxAdd(self,s):
        # discard the empty one, to be left with the refraction string
         sphereParsed   = False
         cylinderParsed = False
         axisParsed     = False
         ax             = np.nan
         sph            = np.nan
         cyl            = np.nan

         try:
             fullStr = s.split('add')[0]
             # break fullStr into sphere cylinder axis and add
             # find the brackets represnenting cylinder and axis
             indStart = fullStr.find('(')
             indEnd   = fullStr.find(')')
             if indStart!=-1 and indEnd!=-1:
                 sphStr   = fullStr[:indStart]
                 sph = 0.0 if len(sphStr)==0 else np.float(sphStr.replace(',','.'))
                 sphereParsed   = True
                 cylAx          = fullStr[indStart+1:indEnd].split('à')
                 cyl            = -np.abs(np.float(cylAx[0].replace(',','.')))
                 cylinderParsed = True
                 ax = np.float(cylAx[1].replace('°',''))
                 axisParsed     = True
             else:
                 sph = np.float(fullStr.replace(',','.'))
                 sphereParsed   = True
                 cyl = 0.0
                 cylinderParsed = True
                 ax = 0.0
                 axisParsed     = True
         except:
             if not sphereParsed:
                 sph = np.nan
             if not cylinderParsed:
                 cyl = np.nan
             if not axisParsed:
                 ax = np.nan

         return sph, cyl, ax

    def _ParseObjectiveRefraction(self,objRefStr):
        # passed in: dataframe
#        vals = pd.DataFrame()
        sph = np.nan*np.ones(2)
        cyl = np.nan*np.ones(2)
        ax  = np.nan*np.ones(2)
        side   =['Left','Right']
        if len(objRefStr)>0:
            for s in range(2):
                objSph = objRefStr['AutoRefractometerSphere_'+side[s]]
                objCyl = objRefStr['AutoRefractometerCylinder_'+side[s]]
                objAx  = objRefStr['AutoRefractometerAxis_'+side[s]]
                if pd.isnull(objSph)==False:
                    objSph = objRefStr['AutoRefractometerSphere_'+side[s]].replace(',','.').replace(' ','')
                    if objSph=='PLAN':
                        objSph = 0.0
                    try:
                        sph[s]   = np.float(objSph)
                    except:
                        sph[s]    = np.nan
                else:
                    sph[s] = np.nan

                if pd.isnull(objCyl)==False:
                    objCyl = objRefStr['AutoRefractometerCylinder_'+side[s]].replace(',','.').replace(' ','')
                    try:
                        cyl[s]    = -np.abs(np.float(objCyl))
                    except:
                        cyl[s]    = np.nan
                else:
                    cyl[s] = np.nan

                if pd.isnull(objAx)==False:
                    objAx  = objRefStr['AutoRefractometerAxis_'+side[s]].replace(',','.').replace(' ','')
                    try:
                        ax[s]    = np.float(objAx)
                    except:
                        ax[s]    = np.nan
                else:
                    ax[s] = np.nan

        sphLeft  = sph[0]
        sphRight = sph[1]
        cylLeft  = cyl[0]
        cylRight = cyl[1]
        axLeft   = ax[0]
        axRight  = ax[1]
        return sphLeft,sphRight, cylLeft, cylRight, axLeft, axRight

    def _ParseRefraction(self,refStr):
        # parse refraction of current glasses from string
        # break the string accorind to end-of-line character
        # subjRefStr is an array of strings

        # Prepare string for parsing
        refStr  = re.sub(r'\d+\s*/\s*\d+','',refStr) # remove patterns of the form e.g. 9/10 or 9 / 10
        refStr  = refStr.lower().replace(';',' ').replace(' ','').replace(',','.').split('\n')

        sphRight = np.nan
        cylRight = np.nan
        axRight  = np.nan
        sphLeft  = np.nan
        cylLeft  = np.nan
        axLeft   = np.nan

        rightParsed = False
        leftParsed  = False
        for rIdx in refStr:
            # Try parsing right eye data
            rIdx = re.sub('[=]?[aA]dd.*','',rIdx)
            if not rightParsed:
                m = re.match('(od|o.droit)[:=]?',rIdx)
                if m is not None:
                    rIdx = re.sub('(od|o.droit)[:=]?','',rIdx)
                    # parse right
                    sphRight, cylRight, axRight = self._ExtractSphCylAxAdd(rIdx)
                    rightParsed = True

            # Try parsing left eye data
            if not leftParsed:
                m = re.match('(og|o.gauche)[:=]?',rIdx)
                if m is not None:
                    # parse left
                    rIdx = re.sub('(og|o.gauche)[:=]?','',rIdx)
                    sphLeft, cylLeft, axLeft = self._ExtractSphCylAxAdd(rIdx)
                    leftParsed = True

            if not np.isnan(sphRight) and not np.isnan(cylRight) and not np.isnan(axRight):
                rightParsed = True
            if not np.isnan(sphLeft) and not np.isnan(cylLeft) and not np.isnan(axLeft):
                leftParsed = True

        return sphLeft, sphRight, cylLeft, cylRight, axLeft, axRight

    @staticmethod
    def _ParseRefValFromStr(string):
        """
         A method to translate refraction value string (S,C,A) to real number
        """
        ref_out = np.nan
        if isinstance(string,str):
            string= string.replace(",",".").replace(" ","")
            f = re.findall('[+-]?\d+\.\d+|\d+',string)
            if len(f)==1:
                ref_out = float(f[0])

        return ref_out

    def _ParseKeratometry(self,kerStr,n_c=1.3375)->tuple:
        '''
            Parse keratometry values K1, and K2, from string

            Parameters:
            -----------
            kerStr (str):
              string with the keratometry values
            n_c (float): optional, default=1.3375
             the refractive index of the cornea

             Returns:
             ---------
             k1_Left, k1_Right, k2_Left, k2_Right (float)
              keratometry K1, K2 (in diopter) for the two main meridians
              for right and left eye

        '''


        kerStr   = kerStr.lower().replace(',','.').replace(' ','').split('\n')

        k1_Left  = np.nan
        k2_Left  = np.nan
        k1_Right = np.nan
        k2_Right = np.nan
        valsR    = np.ones(2)*np.nan
        valsL    = np.nan*np.ones(2)
        for kIdx in kerStr:
            format1 = False
            format2 = False
            if kIdx.find('od:rayons:')!=-1:
                # format 2, Right
                key     = 'od:rayons:'
                valsR   = kIdx.split(key)[1].split('moyenne')[0].replace('mm','').split('/')
                format2 = True
            elif kIdx.lower().find('dioptrie')!=-1:
                # format 1, Right eye
                format1 = True
                valsR   = re.findall(r'\d+.?\d+/\d+.?\d+',kIdx)
                if len(valsR)>0:
                    valsR = valsR[0].split('/')
                else:
                    valsR = np.nan*np.ones(2)
             # assign K1 K2
            try:
                for vIdx in  range(len(valsR)):
                    valsR[vIdx] = np.float(valsR[vIdx])
            except:
                valsR = np.nan*np.ones(2)
            if format1:
                k2_Right = np.max(valsR)
                k1_Right = np.min(valsR)
            elif format2:
                k2_Right = 1000*(n_c-1)/np.min(valsR)
                k1_Right = 1000*(n_c-1)/np.max(valsR)

            format1 = False
            format2 = False
            # Left eye
            if kIdx.find('og:rayons:')!=-1:
                # format 2, Left eye
                format2 = True
                key     = 'og:rayons:'
                valsL   = kIdx.split(key)[1].split('moyenne')[0].replace('mm','').split('/')
            elif kIdx.find('dioptrie')!=-1:
                # format 1, Left eye
                format1 = True
                valsL   = re.findall(r'\d+.?\d+/\d+.?\d+',kIdx)
                if len(valsL)>0:
                    valsL = valsL[0].split('/')
                else:
                    valsL = np.nan*np.ones(2)
             # assign K1 K2
            try:
                for vIdx in  range(len(valsL)):
                    valsL[vIdx] = np.float(valsL[vIdx])
            except:
                valsL = np.nan*np.ones(2)
            if format1:
                k2_Left = np.max(valsL)
                k1_Left = np.min(valsL)
            elif format2:
                k2_Left = 1000*(n_c-1)/np.min(valsL)
                k1_Left = 1000*(n_c-1)/np.max(valsL)

        return k1_Left, k1_Right, k2_Left, k2_Right

    def _ParsePachimetry(self, pacStr):
        pacStr   = pacStr.lower().replace(' ','').replace(',','.').split('\n')
        pacLeft  = np.nan
        pacRight = np.nan
        for pIdx in pacStr:
            if pIdx.find('od')!=-1:
                try:
                    m = re.findall(r'od.*=.*\d+.*µ?',pIdx.lower())
                    if len(m)>0:
                        pacRight = np.float(re.findall(r'\d+',m[0])[0])
                except:
                    pacRight = np.nan

            elif pIdx.find('og')!=-1:
                try:
                    m = re.findall(r'og.*=.*\d+.*µ?',pIdx.lower())
                    if len(m)>0:
                        pacLeft = np.float(re.findall(r'\d+',m[0])[0])
                except:
                    pacLeft     = np.nan
        return pacLeft, pacRight

    def ParseFromMDB(self,mdb_file,csv_output_folder):
        """
         Run a full path from mdb to parsing

         Parameters:
         -----------
            mdb_folder (str): a path to the folder containing the mdb files of the EMR
        """
        self.MDB2CSV(mdb_file, csv_output_folder)
        # from the output folder, load the required csv files for parsing, namely, Patients, consultation and refraction
        self.Load(os.path.join(csv_output_folder,'Patients.csv'),
                  os.path.join(csv_output_folder,'tREFRACTION.csv'),
                  os.path.join(csv_output_folder,'Consultation.csv')
                  )
        self.Parse()

        return 0

    @staticmethod
    def MDB2CSV(mdb_file, output_folder,encoding='utf8'):
        """
            reads an mdb database, export all tables to designated folder as csv files

            Parameters:
            ------
            mdb_file (str):
                a path to the mdb file
            output_folder (str):
                a path to the output folder, the exported tables will have the same names

            Returns:
            ---------
            exports csv tables to designated folder

        """
        # Check that mdb-tools is in the system
        try:
            spo = subprocess.run('mdb-tables',capture_output=True)
            mdb_tables_exist = True
        except:
            mdb_tables_exist = False

        try:
            spo = subprocess.run('mdb-export',capture_output=True)
            mdb_export_exist = True
        except:
            mdb_export_exist = False

        if not mdb_export_exist:
            raise ValueError('mdb-export cannot be found. is mdb-tools installed?')
        if not mdb_tables_exist:
            raise ValueError('mdb-tables cannot be found. is mdb-tools installed?')

        t = subprocess.run(['mdb-tables',mdb_file],capture_output=True,encoding=encoding)
        # break table names
        # export the needed tables for refraction
        tables = t.stdout.replace("\n","").split(" ")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        for tIdx in tables:
            # Export tables to csv
            print(f'--------Exporting {tIdx}--------')
            table_out = subprocess.run(['mdb-export', f"{mdb_file}",tIdx],encoding=encoding,capture_output=True)
            if table_out.returncode==0:
                with open(f'{os.path.join(output_folder,tIdx)}.csv','w') as file:
                    file.write(table_out.stdout)

    @staticmethod
    def _ResolveAgeByDates(birth_date,exam_date,dayfirst=False,yearfirst=False):
        """
         Infer from the examdate and birthdate, the age of the patient
         at time of examination.
         Return a string for the corrected full format of the birthdate and examination date
         and return the age in years

         Parameters:
         -------------
         birth_date : (str)
            string representing the birth date
         exam_date : (str)
           string representing the examination date

         Returns:
         ------------
          bdate_str : (str)
            string representing the birth date
          edate_str : (str)
            string representing the examination date
          age : (float)
            age in years
        """
        if isinstance(birth_date,str)&isinstance(exam_date,str):
            try:
                birth_date = re.findall("\d*/\d*/\d*",birth_date)
                exam_date  = re.findall("\d*/\d*/\d*",exam_date)

                if len(birth_date)==1:
                    birth_date=birth_date[0]
                    flag = True
                else:
                    flag = False
                if len(exam_date)==1:
                    exam_date=exam_date[0]
                    flag = flag&True
                else:
                    flag = flag&False

                if flag:
                    b_date = dateutil.parser.parse(birth_date,dayfirst = dayfirst, yearfirst=yearfirst)
                    e_date = dateutil.parser.parse(exam_date, dayfirst = dayfirst, yearfirst=yearfirst)
                    # first make sure that the year of the examination does not exceed the current year
                    if e_date> datetime.datetime.now():
                        e_date = e_date.replace(year=e_date.year-100)
                    # then make sure the birth date does not exceed the examination date
                    if b_date.year>e_date.year:
                        b_date = b_date.replace(year=b_date.year-100)
                    if e_date>b_date:
                        age       = (e_date-b_date).days/365
                        edate_str = f'{e_date.day:02f}/{e_date.month:02f}/{e_date.year}'
                        bdate_str = f'{b_date.day:02f}/{b_date.month:02f}/{b_date.year}'
                    else:
                        age       = np.nan
                        edate_str = np.nan
                        bdate_str = np.nan


                    return bdate_str, edate_str, age
                else:
                    # print(f'either exam date or birth date format is incorrect bdate={birth_date}, edate={exam_date}')
                    return np.nan, np.nan, np.nan
            except:
                # print(f"Failed with bdate={birth_date} and edate={exam_date}")
                return np.nan, np.nan, np.nan
        else:
            # print(f'birth date or exam date are not string bdate={birth_date}, edate={exam_date}')
            return np.nan, np.nan, np.nan

    @staticmethod
    def MergeWithVx130DB(emr_db,vx_db):
        """Merge the EMR with the visionix vx130 DB

        Parameters:
        -----------
            emr_db (str):
                a path to the emr csv DB file
            vx_db (str):
                a path to the vx130 csv DB file
                make sure the vx130 has been transformed prior to matching
                see vx120Transformer class

        Returns:
        ---------
            jointDB : DataFrame
                joint EMR-vx130 database
        """
        # dataFolder = '/home/ofir/Work/EyeLib/Data/EMR/Prevot/'
        print('[Merger] Loading VX data')
        vx         = pd.read_csv(vx_db,low_memory=False)
        if not '_isTransformed' in vx.keys():
            print(f'The vx dataset is not transformed. Please use vx120Transformer.Transform() from vx120Transformer class prior to merging')
            return 1

        print('[Merger] Loading EMR data')
        emr        = pd.read_csv(emr_db,low_memory=False)
        # Parse data
        emr_keys    = ['FirstName','Surname','ExamDate','BirthDate'] # keys which make up the ID
        vx_keys     = ['Firstname','Surname','CurrentDate','BirthDate']
        # Remove empty rows
        emrData  = emr.loc[emr[emr_keys].notna().all(axis=1),:]

        # make emr ID
        # emrID  = (emrData['FirstName']+emrData['Surname']+emrData['ExamDate']).str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        emr_name    = emrData['FirstName'].str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        emr_surname = emrData['Surname'].str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        emr_eDate   = emrData['ExamDate'].str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        emr_bDate   = emrData['BirthDate'].str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        emrID       = emr_name+emr_surname+emr_eDate

        vxData = vx.loc[vx[vx_keys].notna().all(axis=1),:]
        # make vx ID 1
        # vxID    = (vxData['Firstname']+vxData['Surname']+vxData['ExamDate']).str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        vx_name    = vxData['Firstname'].str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        vx_surname = vxData['Surname'].str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        vx_eDate   = vxData['ExamDate'].str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()
        vx_bDate   = vxData['BirthDate'].str.replace('[^a-zA-Z0-9]+','',flags=re.IGNORECASE).str.lower()

        vxID       = vx_name+vx_surname+vx_eDate

        #%% Match
        indexVX       = []
        indexEMR      = []
        unmatchedInds = []
        unmatched_reason = []
        NUMCASES      = len(vxData)
        for vIdx in range(len(vxData)):
            # eInds = emrData.loc[emrID==vxID.iloc[vIdx]].index
            name_ind    = emr_name==vx_name.iloc[vIdx]
            surname_ind = emr_surname==vx_surname.iloc[vIdx]
            eDate_ind   = emr_eDate==vx_eDate.iloc[vIdx]
            bDate_ind   = emr_bDate==vx_bDate.iloc[vIdx]

            eInds       = emrID.loc[name_ind&surname_ind&eDate_ind].index

            print(f'[Merger] {vIdx+1}/{NUMCASES} found {len(eInds)} matches')

            for eIdx in eInds:
                indexEMR.append(eIdx)
                indexVX.append(vxData.index[vIdx])
            if len(eInds)==0:
                best_match  = name_ind.astype(int)+surname_ind.astype(int)+eDate_ind.astype(int)+bDate_ind.astype(int)
                bm_inds     = np.where(best_match>=3)[0]
                eInds        = emrData.iloc[bm_inds].index
                print(f"[Merger] Best match for {vx_name.iloc[vIdx]} {vx_surname.iloc[vIdx]} is:\n {emrData.loc[eInds][['FirstName','Surname']]}")
                # append the bext match indices (risky.. )
                if len(eInds)>0:
                    for eIdx in eInds:
                        indexEMR.append(eIdx)
                        indexVX.append(vxData.index[vIdx])
                else:
                    unmatchedInds.append(vxData.index[vIdx])
                    uInd = np.zeros(3,dtype=int)
                    if name_ind.sum()==0:
                        uInd[0]=0
                    else:
                        uInd[0]=1
                    if surname_ind.sum()==0:
                        uInd[1] = 0
                    else:
                        uInd[1] = 1
                    if eDate_ind.sum()==0:
                        uInd[2] = 0
                    else:
                        uInd[2] = 1
                    unmatched_reason.append(uInd)

        vxDataNew        = vxData.loc[indexVX,:]
        emrDataNew       = emrData.loc[indexEMR,:]
        emrDataNew.index = range(len(emrDataNew))# np.arange(0,len(emrDataNew),1)
        vxDataNew.index  = range(len(vxDataNew))# np.arange(0,len(vxDataNew),1)
        jointDB          = vxDataNew.copy()
        # indicate columns names with prefix EMR: as coming from the EMR
        for cIdx in emrDataNew.columns:
            jointDB.loc[:,f'EMR:{cIdx}'] = emrDataNew[cIdx]

        return jointDB, unmatchedInds,unmatched_reason