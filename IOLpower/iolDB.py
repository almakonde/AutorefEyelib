import dateutil
import datetime
import pandas as pd
import numpy as np
import re

class DataBase:
    """
     IOL power database related codes
    """
    def CreateDBs(self,OplusDB,export=False):
        """
         NOW OBSOLETE
         Create three db from the oplus db preop; opday and post op
         the OplusDB needs to be the parsed one (see OplusEMRParser.Parser)

         Parameters:
         -----------
          OplusDB, str
            a path to the Oplus parsed db in csv format
         export, bool, default = False
           export the joint db to csv format
        """

        print("Loading Oplus data")
        data = pd.read_csv(OplusDB,low_memory=False)
        # for each index in left, search for pre and post operative operations
        # remove all lines where the Exam date is not recorded
        validEdate   = data['ExamDate'].isna()==False
        validBdate   = data['BirthDate'].isna()==False
        validName    = data['FirstName'].isna()==False
        validSurname = data['Surname'].isna()==False
        data         = data.loc[data.index[validName&validSurname&validEdate&validBdate]]

        # find all lines with either left or right eye IOL implant procedure
        indsLeft  = np.where(data['IolPower_Left'].notna())[0]  # inidicator of surgery
        indsRight = np.where(data['IolPower_Right'].notna())[0]

        # parse examDate
        examDate  = np.ndarray(shape = len(data), dtype=datetime.datetime)
        patID     = np.ndarray(shape= len(data),dtype = list)
        for eIdx in range(len(data)):
            # record the examination dat
            examDate[eIdx] = dateutil.parser.parse(data.loc[data.index[eIdx],'ExamDate'])
            patID[eIdx]    = str(data.loc[data.index[eIdx],'FirstName'].replace(' ','').lower()+data.loc[data.index[eIdx],'Surname'].replace(' ','').lower()+data.loc[data.index[eIdx],'BirthDate'])

        #%% find patients
        preop  = pd.DataFrame() # pre operation
        postop = pd.DataFrame() # post operation
        opday  = pd.DataFrame() # day of operation
        temp   = pd.DataFrame(data= np.nan*np.ones(shape=(1,len(data.keys()))),columns = data.keys())
        # we treat the preop as all examination up to and including the day of operation
        for il in indsLeft:
            # find all pre op examination for this particular patient
            opday         = opday.append(data.loc[data.index[il]])
            patInds       = np.where(patID == patID[il])[0]
            print(f"Patient {patID[il]}: found {len(patInds)} matches")
            exams         = examDate[patInds] # all the examination of the current patient
            preExamsInds  = patInds[exams<examDate[il]]
            postExamsInds = patInds[exams>examDate[il]]
            rowScore      = data.loc[data.index[patInds]].notna().sum(axis=1)/len(data.keys()) -abs(data.index[patInds]-data.index[il])/len(patInds)
            # find the last exam before current and treat it as pre op
            if len(preExamsInds)>0:
                preopInds = data.index[patInds[exams<examDate[il]]]
                preop    = preop.append(data.loc[preopInds[np.argsort(rowScore[preopInds].values)[-1]]],ignore_index=False)
                # preopInd = preExamsInds[np.argsort(exams[exams<examDate[il]])[-1]]
                # preop    = preop.append(data.loc[data.index[preopInd]])
            else:
                preop     = preop.append(temp)
            if len(postExamsInds)>0:
                postopInds = data.index[patInds[exams>examDate[il]]]
                postop     = postop.append(data.loc[postopInds[np.argsort(rowScore[postopInds].values)[-1]]],ignore_index=False)
            else:
                postop  = postop.append(temp)

        for ir in indsRight:
            # find all pre op examination for this particular patient
            opday         = opday.append(data.loc[data.index[ir]])
            patInds       = np.where(patID == patID[ir])[0]
            print(f"Patient {patID[ir]}: found {len(patInds)} matches")
            exams         = examDate[patInds] # all the examination of the current patient
            preExamsInds  = patInds[exams<examDate[ir]]
            postExamsInds = patInds[exams>examDate[ir]]
            rowScore      = data.loc[data.index[patInds]].notna().sum(axis=1)/len(data.keys()) - abs(data.index[patInds]-data.index[ir])/len(patInds)
            # find the last exam before current and treat it as pre op
            if len(preExamsInds)>0:
                preopInds = data.index[patInds[exams<examDate[ir]]]
                preop     = preop.append(data.loc[preopInds[np.argsort(rowScore[preopInds].values)[-1]]],ignore_index=False)
                # preopInd = preExamsInds[np.argsort(exams[exams<examDate[ir]])[-1]]
                # preop    = preop.append(data.loc[data.index[preopInd]])
            else:
                preop   = preop.append(temp)
            if len(postExamsInds)>0:
                postopInds = data.index[patInds[exams>examDate[ir]]]
                postop    = postop.append(data.loc[postopInds[np.argsort(rowScore[postopInds].values)[-1]]],ignore_index=False)
                # postopInd = postExamsInds[np.argsort(exams[exams>examDate[ir]])[0]]
                # postop    = postop.append(data.loc[data.index[postopInd]])
            else:
                postop  = postop.append(temp)
        if export:
            preop.to_csv("PreOp.csv",index=False)
            postop.to_csv("PostOp.csv",index=False)
            opday.to_csv("OpDay.csv",index=False)

        return preop, opday,postop

    def MergeIOLMasterOplus(self,iolData,emrData): # obsolete
        """
          Match iolMaster database to Oplus EMR database
          Parameters:
          ------------
          iolData, DataFrame,
           A DataFrame with iolMaster raw data
          emrData, DataFrame
           a parsed Oplus emr databse
           the correct format of the emrData can be obtained with the Oplus parser code
           Teh Oplus data should contain fields related to pre and post IOL operation
           such as implanted power, final refraction, the A constant, target refraction
           and the formula used to compute the IOL power durig preoperative planning

           Patients are matched based on surname+firstname+birthdate
        """
        # alist of a constants and iol name
        # the dicrtionary below contain manufacturer A-const.
        # optimized A-const should be obtained by retrospective analysis of the data

        aconst = {'medicontur biflex multifocal': 118.9,
                'lentis confomrt':118,
                'ma60ac': 118.4,
                'asphina_5009m':118.3
                }

        # set a list of features required for the post op data
        requiredPostopFields  = ['VisualAcuitySphere_Right','VisualAcuitySphere_Left',
                                'VisualAcuityCylinder_Right','VisualAcuityCylinder_Left',
                                'VisualAcuityAxis_Right','VisualAcuityAxis_Left',
                                'AutoRefractometerSphere_Right','AutoRefractometerSphere_Left',
                                'AutoRefractometerCylinder_Right','AutoRefractometerCylinder_Left',
                                'AutoRefractometerAxis_Right','AutoRefractometerAxis_Left'] # required postop fields
        requiredPreOpFields = ['IolDesiredRefractionPostOp','IolFormula',
                                'IolFinalRefraction','IolPower','IolTheoreticalPower',
                                'IolID','IolMeanK','IolAxialLength',
                                'IolModel','IolModelFamily',]# related to surgery planning and IOL formulas
        # Create two iolMaster IDs, one with day/month/year and one with month/day/year, in order
        # to overcome issues in dateparsing in the Oplus EMR
        iolBdates     = iolData['patients_birth_date'].apply(self._SplitDateStr, args=[True]) # format [day, month, year], dtype=str
        iolBdates     = np.asanyarray(iolBdates.to_list())
        iolID1        = iolData[['last_name','first_name']]
        iolID1.loc[:,'Day']   = iolBdates[:,0]
        iolID1.loc[:,'Month'] = iolBdates[:,1]
        iolID1.loc[:,'Year']  = iolBdates[:,2]
        iolID1 = (iolID1.sum(axis=1).dropna().str.lower()
                 .apply(self._SimplifiedString)
                 .apply(self._SimplifiedFrench)
                 )
        iolID2                = iolData[['last_name','first_name']]
        iolID2.loc[:,'Day']   = iolBdates[:,1] # switch day and month
        iolID2.loc[:,'Month'] = iolBdates[:,0]
        iolID2.loc[:,'Year']  = iolBdates[:,2]
        iolID2 = (iolID2.sum(axis=1).dropna().str.lower()
                 .apply(self._SimplifiedString)
                 .apply(self._SimplifiedFrench)
                 )

        emrData = emrData.loc[emrData[['FirstName','Surname','BirthDate','ExamDate']].dropna().index]
        emrID   = emrData[['Surname','FirstName','BirthDate']]
        emrID   = (emrID.sum(axis=1).dropna().str.lower()
                  .apply(self._SimplifiedString)
                  .apply(self._SimplifiedFrench)
                  )
        emrData = emrData.loc[emrID.index]
        chi     = emrData['ChirurgicalAct'].apply(self._ParseChirurgicalAct)
        # Add IOLpowerImplanted Left and Right
        chi = np.asanyarray(chi.to_list())
        emrData.loc[emrData.index,"IolPowerImplanted_Right"] = chi[:,0]
        emrData.loc[emrData.index,'IolImplanted_Right']      = chi[:,1]
        emrData.loc[emrData.index,"IolPowerImplanted_Left"]  = chi[:,2]
        emrData.loc[emrData.index,'IolImplanted_Left']       = chi[:,3]
        # extract only
        emrData.set_index(emrID,inplace=True)

        jointDB     = pd.DataFrame(columns=list(emrData.columns)+list(iolData.columns))
        # postOpDB    = pd.DataFrame(columns=list(emrData.columns))
        nextNoDate  = 0

        nextRow              = 0
        nextNoDate           = 0
        numMatchedPatients   = 0
        numUnmatchedPatients = 0
        numFaulty            = 0
        numLeft = 0
        numRight =0
        for pIdx in range(len(iolID1)):
            emr_inds1 = emrID[emrID==iolID1.loc[iolID1.index[pIdx]]].index
            emr_inds2 = emrID[emrID==iolID2.loc[iolID2.index[pIdx]]].index
            if len(emr_inds1)>0:
                emr_inds = emr_inds1
                iolIdx   = iolID1.index[pIdx]
            elif len(emr_inds2)>0:
                emr_inds = emr_inds2
                iolIdx   = iolID2.index[pIdx]
            else:
                emr_inds = []

            if len(emr_inds)>0:
                # sort emrData by examination date
                eData = emrData.loc[emr_inds]
                eDate = pd.to_datetime(eData.loc[:,'ExamDate'].dropna(),dayfirst=True).sort_values()
                eData = eData.loc[eDate.index]

                # Parse iol measurement date
                iDate   = pd.to_datetime(iolData.loc[iolIdx,'measurement_date'],dayfirst=True)
                chiDate = eDate[eData[['IolPowerImplanted_Right','IolPowerImplanted_Left']].notna().sum(axis=1)>=1]
                # find the first chi larger than iDate
                try:
                    dateDiff = np.where(np.asanyarray(chiDate-iDate,dtype=float)>=0)[0]
                except:
                    dateDiff= []
                if len(dateDiff)==1: # only one chi date is higher
                    # from preop take the surgery planning
                    preop  = eData.loc[:chiDate.index[dateDiff[0]]].drop(index=chiDate.index[dateDiff[0]])
                    # in the preop, find the row for which the surgery planning was made
                    preopInds = preop.index[np.where(preop[['IolFormula_Right','IolFormula_Left']].notna().sum(axis=1)>=1)[0]]
                    preop     = preop.loc[preopInds[-1]] if len(preopInds)>0 else None

                    # from opday take the power implanted
                    opday    = eData.loc[chiDate.index[dateDiff[0]]]
                    # frompostop take refraction
                    postop   = eData.loc[chiDate.index[dateDiff[0]]:].drop(index=chiDate.index[dateDiff])
                    # choose the post op data with the most ammount of data
                    rowRanks = postop[requiredPostopFields].notna().sum(axis=1).values/len(requiredPostopFields)
                    if len(rowRanks)>1:
                        rowInd   = np.argsort(rowRanks)[-1]
                        if rowRanks[rowInd]>0:
                            postop = postop.loc[postop.index[rowInd]]
                        else:
                            postop = None
                    else:
                        postop = None
                    # eData    = eData.loc[chiDate.index[dateDiff[0]]:]
                elif len(dateDiff)>1: # for several surgeries after iol master examination date
                    preop  = eData.loc[:chiDate.index[dateDiff[0]]].drop(index=chiDate.index[dateDiff[0]])
                    # in the preop, find the row for which the surgery planning was made
                    preopInds = preop.index[np.where(preop[['IolFormula_Right','IolFormula_Left']].notna().sum(axis=1)>=1)[0]]
                    preop     = preop.loc[preopInds[-1]] if len(preopInds)>0 else None

                    # Get surgery date associated data
                    opday  = eData.loc[chiDate.index[dateDiff[0]]]

                    # Get post op data
                    pInds  = (eData.index>=min(chiDate.index[dateDiff[[0,1]]]))&(eData.index<=max(chiDate.index[dateDiff[[0,1]]]))
                    postop = eData.loc[pInds].drop(index=chiDate.index[dateDiff[:2]])
                    # choose the post op data with the most ammount of data
                    rowRanks = postop[requiredPostopFields].notna().sum(axis=1).values/len(requiredPostopFields)
                    if len(rowRanks)>1:
                        rowInd   = np.argsort(rowRanks)[-1]
                        if rowRanks[rowInd]>0:
                            postop = postop.loc[postop.index[rowInd]]
                        else:
                            postop = None
                    else:
                        postop = None
                elif len(dateDiff)==0:
                    opday  = None
                    preop  = None
                    postop = None

                    # eData  = None
                # Process the EMR data found for each iolMaster measurement
                if opday is not None:
                    eye = []
                    if pd.notna(opday['IolPowerImplanted_Right']):
                        eye.append('Right')
                        numRight +=1
                    if pd.notna(opday['IolPowerImplanted_Left']):
                        eye.append('Left')
                        numLeft+=1
                    print(f'[iolDB] Found OpDay data for eye ={eye} ({pIdx}/{len(iolData)})')
                    # from opday take all data
                    jointDB.loc[nextRow,opday.keys()]  = opday
                    # add the iolMaster data
                    jointDB.loc[nextRow,iolData.keys()] = iolData.loc[iolIdx]

                    if preop is not None:
                        # Add the preopdata Data, desired refraction iol formula and predicted refraction
                        jointDB.loc[nextRow,'PreopDate'] = preop['ExamDate']
                        for fIdx in requiredPreOpFields:
                            for eIdx in eye:
                                vals = preop[fIdx+'_'+eIdx]
                                # if len(vals)>=1:
                                print(f'[iolDB] Found Preop data for {pIdx}/{len(iolData)}')
                                # vals = vals.drop(index=vals.index[:-1]).values[0]
                                jointDB.loc[nextRow,fIdx+'_'+eIdx] = vals
                                # else:
                                #     print(f'[iolDB] No values for {fIdx+"_"+eIdx}  in Preop data found for patient ({pIdx}/{len(iolData)})')
                    else:
                        print(f'[iolDB] No Preop data for ({pIdx}/{len(iolData)})')

                    if postop is not None:
                        # populate postop fields
                        print(f'[iolDB] Found Postop data for ({pIdx}/{len(iolData)})')
                        jointDB.loc[nextRow,requiredPostopFields] = postop[requiredPostopFields]
                        jointDB.loc[nextRow,'PostOpDate'] = postop['ExamDate']
                        # compute followup time
                        followupDiff = pd.to_datetime(postop['ExamDate'],dayfirst=True)-pd.to_datetime(jointDB.loc[nextRow,'ExamDate'],dayfirst=True)
                        jointDB.loc[nextRow,'Followup_Days'] = followupDiff.days

                    else:
                        print(f'[iolDB] No Postop data for ({pIdx}/{len(iolData)})')
                    nextRow+=1
                    numMatchedPatients+=1
                else:
                    # In the case of no existing EMR record after the iolMaster measurement date
                    print(f'[iolDB] No surgery data for ({pIdx}/{len(iolData)})')
                    nextNoDate+=1
                    numFaulty+=1
            else:
                print(f'[iolDB] No EMR matches for ({pIdx}/{len(iolData)})')
                numUnmatchedPatients+=1

        print(f'[iolDB] Total number of patients matched={numMatchedPatients}, unmatched={numUnmatchedPatients}, no surgery date={numFaulty}')
        print(f'[iolDB] Num. left eye matched={numLeft}, Num. right eye matches={numRight}')
        return jointDB#,postOpDB

    @staticmethod
    def _ParseChirurgicalAct(chiStr, maxPower=33, minPower=10):
        """
         Extract the power of a lens implanted and the eye
         Parameters:
         -------------
         chiStr, str
          string containing chirurgical act info
         maxPower, float
          maximal power allowwed for IOL
         minPower, float
          minimal power allowed for IOL
        """
        # TODO: move to Oplus parser
        # iol output organized as [iolPower_Right, iolName_right, iolPower_Left, iolName_Left]
        iol = [pd.NA, pd.NA,pd.NA, pd.NA]
        if isinstance(chiStr,str):
            chiStr = chiStr.lower()
            if len(re.findall(r'phk|icp',chiStr,flags=re.IGNORECASE))>0:
                p = re.findall(r'[+-]?\_?\d+\.\d+|[+-]?\_?\d+',chiStr)
                if len(p)>0:
                    # search the number most suited to represent the IOL power
                    iolPower_temp = []
                    iolPower      = None
                    for pIdx in range(len(p)):
                        iolPower_temp = float(p[pIdx].replace(' ','').replace('+',''))
                        if (iolPower_temp>=minPower)&(iolPower_temp<=maxPower):
                            iolPower = iolPower_temp
                            iolpInd  = pIdx

                    if iolPower is not None:
                        # Extract IOL power and model
                        # Right eye
                        if len(re.findall(r'od',chiStr,flags=re.IGNORECASE))>0:
                            iol[0]   = iolPower
                            # find the lens name
                            strStart = chiStr.find('od')
                            strEnd   = chiStr.find(p[iolpInd])
                            iol[1]   = chiStr[strStart+3:strEnd-1]
                            pu        = re.findall(r'puiss[:]?',chiStr,flags=re.IGNORECASE)
                            if len(pu)>0:
                                iol[1] = iol[1].replace(pu[0],'').replace('puiss','')
                        # Left eye
                        if len(re.findall(r'og',chiStr,flags=re.IGNORECASE))>0:
                            iol[2]   = iolPower
                            # find the lens name
                            strStart = chiStr.find('og')
                            strEnd   = chiStr.find(p[iolpInd])
                            iol[3]   = chiStr[strStart+3:strEnd-1]
                            pu        = re.findall(r'puiss[:]?',chiStr,flags=re.IGNORECASE)
                            if len(pu)>0:
                                iol[3] = iol[3].replace(pu[0],'')
        return iol

    def Merge(self,iolData,emrData,max_measurement_to_op_days = 90,min_followup_days=15,max_followup_days=120):
        """
          Match iolMaster database to Oplus EMR database
          Patients are matched based on surname+firstname+birthdate

          Parameters:
          ------------
          iolData, DataFrame,
           A DataFrame with iolMaster raw data
          emrData, DataFrame
           A parsed Oplus emr databse
           the correct format of the emrData can be obtained with the Oplus parser code
           Teh Oplus data should contain fields related to pre and post IOL operation
           such as implanted power, final refraction, the A constant, target refraction
           and the formula used to compute the IOL power durig preoperative planning
          max_measurement_to_op_days, int, default =90
            maximal number of days between iol master measurement and the surgery date
          min_fillowup_days, int, default=15
            minimal number of days between the surgery and the followup measurements
          max_followup_days, int, default = 90
            mximal numberof days between surgery and followup measurements
          Returns:
          ---------
          jointDB, DataFrame
           a database containing patient data, measurments for surgery day, chosen IOL and its power,
           and followup data. the autorefractometers S, C, A are always for followu data
           the IOL information is preop data, refering to the surgery planing stage, including theoretical
           power, desired refraction post op and predicted refraction from the chosen IOL formula
        """
        # alist of a constants and iol name
        # the dicrtionary below contain manufacturer A-const.
        # optimized A-const should be obtained by retrospective analysis of the data

        aconst = {'medicontur biflex multifocal': 118.9,
                'lentis confomrt':118,
                'ma60ac': 118.4,
                'asphina_5009m':118.3
                }

        # set a list of features required for the post op data
        requiredPostopFields = ['VisualAcuitySphere_Right','VisualAcuitySphere_Left',
                                'VisualAcuityCylinder_Right','VisualAcuityCylinder_Left',
                                'VisualAcuityAxis_Right','VisualAcuityAxis_Left']#,
                                # 'AutoRefractometerSphere_Right','AutoRefractometerSphere_Left',
                                # 'AutoRefractometerCylinder_Right','AutoRefractometerCylinder_Left',
                                # 'AutoRefractometerAxis_Right','AutoRefractometerAxis_Left'] # required postop fields
        requiredPreOpFields  = ['IolDesiredRefractionPostOp','IolFormula',
                                'IolFinalRefraction','IolPower','IolTheoreticalPower',
                                'IolID','IolMeanK','IolAxialLength',
                                'IolModel','IolModelFamily',]# related to surgery planning and IOL formulas
        # Create two iolMaster IDs, one with day/month/year and one with month/day/year, in order
        # to overcome issues in dateparsing in the Oplus EMR
        iolBdates             = iolData['patients_birth_date'].apply(self._SplitDateStr, args=[True]) # format [day, month, year], dtype=str
        iolBdates             = np.asanyarray(iolBdates.to_list())
        iolID1                = iolData[['last_name','first_name']]
        iolID1.loc[:,'Day']   = iolBdates[:,0]
        iolID1.loc[:,'Month'] = iolBdates[:,1]
        iolID1.loc[:,'Year']  = iolBdates[:,2]
        iolID1 = (iolID1.sum(axis=1).dropna().str.lower()
                 .apply(self._SimplifiedString)
                 .apply(self._SimplifiedFrench)
                 )
        iolID2                = iolData[['last_name','first_name']]
        iolID2.loc[:,'Day']   = iolBdates[:,1] # switch day and month
        iolID2.loc[:,'Month'] = iolBdates[:,0]
        iolID2.loc[:,'Year']  = iolBdates[:,2]
        iolID2 = (iolID2.sum(axis=1).dropna().str.lower()
                 .apply(self._SimplifiedString)
                 .apply(self._SimplifiedFrench)
                 )

        emrData = emrData.loc[emrData[['FirstName','Surname','BirthDate','ExamDate']].dropna().index]
        emrID   = emrData[['Surname','FirstName','BirthDate']]
        emrID   = (emrID.sum(axis=1).dropna().str.lower()
                  .apply(self._SimplifiedString)
                  .apply(self._SimplifiedFrench)
                  )
        emrData = emrData.loc[emrID.index]
        chi     = emrData['ChirurgicalAct'].apply(self._ParseChirurgicalAct)
        # Add IOLpowerImplanted Left and Right
        chi = np.asanyarray(chi.to_list())
        emrData.loc[emrData.index,"IolPowerImplanted_Right"] = chi[:,0]
        emrData.loc[emrData.index,'IolImplanted_Right']      = chi[:,1]
        emrData.loc[emrData.index,"IolPowerImplanted_Left"]  = chi[:,2]
        emrData.loc[emrData.index,'IolImplanted_Left']       = chi[:,3]
        emrData.set_index(emrID,inplace=True)
        opDayDB = emrData[emrData[['IolPowerImplanted_Right','IolPowerImplanted_Left']].notna().any(axis=1)]

        jointDB = pd.DataFrame(columns=list(emrData.columns)+list(iolData.columns))
        nextRow = 0
        for pIdx in range(len(opDayDB)):
            # For each surgery, find the matching iolMaster measurement done previous to surgery date
            inds = (iolID1==opDayDB.index[pIdx]) | (iolID2==opDayDB.index[pIdx]) # inds in the iolData
            if inds.sum()>=1:
                print(f'[iolDB] Found match ({pIdx}/{len(opDayDB)})')
                # take all patient data
                eData = emrData.loc[opDayDB.index[pIdx]]
                eDate = pd.to_datetime(eData['ExamDate'],errors='ignore',dayfirst=True)
                # eData = eData.sort_values(by=eDate) # sort by examination date
                # Take the  matching iolMaster data
                iData   = iolData.loc[inds]
                iDate   = pd.to_datetime(iData['measurement_date'],dayfirst=True).sort_values()
                chiDate = pd.to_datetime(opDayDB['ExamDate'].iloc[pIdx],dayfirst=True)
                #  find the iolMaster examination prior to the opDay
                iInds = np.where(iDate<=chiDate)[0]
                if len(iInds)>0:
                    iolMasterToOp_days = (chiDate-iDate.iloc[iInds[-1]]).days
                    if iolMasterToOp_days<=max_measurement_to_op_days:
                        jointDB.loc[nextRow,iolData.keys()] = iData.loc[iData.index[iInds[-1]]]
                        jointDB.loc[nextRow,emrData.keys()] = opDayDB.iloc[pIdx]
                        jointDB.loc[nextRow,'iolMasterToSurgeryDays'] = iolMasterToOp_days
                        # find the surgery planning prior to the surgery date
                        planningDB = eData[eData[['IolFormula_Right','IolFormula_Left']].notna().sum(axis=1)>=1]
                        planDate   = pd.to_datetime(planningDB['ExamDate'],errors='ignore',dayfirst=True)
                        pDatesInds = np.where(planDate<=chiDate)[0]
                        if len(pDatesInds)>=1:
                            preop      = planningDB.iloc[pDatesInds[-1]]
                            # populate the preop fields
                            for fIdx in requiredPreOpFields:
                                for eIdx in ['_Right','_Left']:
                                    jointDB.loc[nextRow,fIdx+eIdx] = preop[fIdx+eIdx]
                        else:
                            print(f'[iolDB] No preOp data for ({pIdx}/{len(opDayDB)})')
                        # organize postop data
                        # first find the chi date in eData, then take examination after
                        diff_days = (eDate-chiDate).apply(pd.Timedelta.total_seconds)/60/60/24
                        postOp    = eData[(diff_days>=min_followup_days)&(diff_days<=max_followup_days)]
                        # postOp  = eData.iloc[np.where(eDate>=chiDate)[0]]
                        if len(postOp)>0:
                            # rank the rows according to availability of required postop fields and take the one with the best score
                            rowRank = postOp[requiredPostopFields].notna().sum(axis=1)/len(requiredPostopFields)
                            if max(rowRank)>0:
                                # bestInd  = np.argmax(rowRank) #TODO: choose either the first or the last one
                                bestInds = np.where(rowRank==np.max(rowRank))[0]
                                # take the latest date
                                bestInd  = np.argmax((postOp['ExamDate'].apply(pd.to_datetime,dayfirst=True)-chiDate).iloc[bestInds].apply(pd.Timedelta.total_seconds).values/60/60/24)
                                postOpDB = postOp.iloc[bestInd]
                                jointDB.loc[nextRow,requiredPostopFields] = postOpDB[requiredPostopFields]
                                jointDB.loc[nextRow,'Followup_Days']      = (pd.to_datetime(postOpDB['ExamDate'],dayfirst=True)-chiDate).days
                            else:
                                print(f'[iolDB] No post op data for ({pIdx}/{len(opDayDB)})')
                        else:
                            print(f'[iolDB] No followup data in specified time range for ({pIdx}/{len(opDayDB)})')
                        nextRow+=1

                    else:
                        print(f'[iolDB] IOLmaster measurement too far from op day ({pIdx}/{len(opDayDB)})')
                else:
                    print(f'[iolDB] No IOLmaster data for ({pIdx}/{len(opDayDB)})')
        return jointDB

    def _SimplifiedFrench(self,stringIn):
        if isinstance(stringIn,str):
            return self._DictReplace(stringIn,{'è':'e',
                                                 'é':'e',
                                                 'ë':'e',
                                                 'ç':'c',
                                                 'ï':'i'})
            # return stringIn.lower().replace('è','e').replace('é','e').replace('ë','e').replace('ç','c').replace('ï','i')
        else:
            return np.nan

    def _SimplifiedString(self,stringIn):
        if isinstance(stringIn,str):
            return self._DictReplace(stringIn,{' ':'',
                                               '-':'',
                                               ':':'',
                                               '/':'',
                                               '!':'',
                                               "'":""})
        else:
            return stringIn

    @staticmethod
    def _DictReplace(strIn,rDict):
        """
         Replace chatacters in a string using a dictionary of replacements
        """
        if isinstance(strIn,str) and isinstance(rDict,dict):
            for kIdx in rDict.keys():
                strIn = strIn.replace(kIdx,rDict[kIdx])
        return strIn

    @staticmethod
    def _DateStrToDatetime(dateStr,dayfirst=True):
        '''Parse a string into datetime format'''
        try:
            l = pd.to_datetime(dateStr,dayfirst=dayfirst).strdtime('')
            l = dateStr.split('/')
            if len(l)==3:
                return datetime.datetime(day=int(l[0]),month=int(l[1]),year=int(l[2]))
            else:
                return np.nan
            # return dateutil.parser.parse(dateStr,dayfirst=dayfirst)
        except:
            return dateStr

    @staticmethod
    def _SplitDateStr(dateStr,dayfirst=True):
        # transform a date string to list of strings [day, month, year]
        try:
            # p = dateutil.parser.parse(dateStr,dayfirst=dayfirst)
            p     = pd.to_datetime(dateStr,dayfirst=dayfirst,errors='raise')
            day   = p.day
            month = p.month
            year  = p.year
            if year>=datetime.datetime.now().year:
               year-=100
            return f'{day:02d}', f'{month:02d}', f'{year:02d}'
        except:
            return np.nan, np.nan, np.nan

    @staticmethod
    def _DateStrToTuple(dateStr,dayfirst=True):
        # transform a date string to list of strings [day, month, year]
        try:
            p = dateutil.parser.parse(dateStr,dayfirst=dayfirst)
            return (f'{p.day:02d}', f'{p.month:02d}', f'{p.year:02d}')
        except:
            return dateStr
