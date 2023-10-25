import numpy as np
import re
import dateutil
import datetime
import pandas as pd

class Parser:
    data = pd.DataFrame() # preallocation

    def Load(self,emrFileName, delim=";", encoding="latin1"):
        # Load EMR csv data and save only required columns
        # Rename the French columns to match English format
        self.emrLoaded = False
        print("[EMR] loading data")
        self.data = pd.read_csv(emrFileName,
                                delimiter=delim,
                                encoding=encoding,  # for French encoding use latin1
                                infer_datetime_format=True,
                                warn_bad_lines=False,
                                low_memory=False,
                                verbose=True,
                                skip_blank_lines=True)
        self.emrLoaded = True
        self.emrParsed = False
        print("[EMR] loading completed")

    def _RenameColumns(self):

        # Construct a dataframe with the needed columns for the refraction project
        # TODO: place the dictionary of fields and name map into a json file
        nameMap = dict({"Prénom du patient":"FirstName",
                        "Nom de famille du patient":"Surname",
                        "Date de naissance":"BirthDate",
                        "Sexe du patient":"Gender",
                        "Date de l\'acte":"ExamDate",
                        "Age du patient pour l\'acte":"Age",
                        "Acte chirurgical":"ChirurgicalAct",
                        "Lampe à fente":"SlitLamp",
                        "Skiascopie":"Skiascopy",
                        "Acte laser":"Laser",
                        "Laser yag":"LaserYag",
                        "Réfractomètre auto.:Sphère OG":"AutoRefractometerSphere_Left",
                        "Réfractomètre auto.:Sphère OD":"AutoRefractometerSphere_Right",
                        "Réfractomètre auto.:Cylindre OG":"AutoRefractometerCylinder_Left",
                        "Réfractomètre auto.:Cylindre OD":"AutoRefractometerCylinder_Right",
                        "Réfractomètre auto.:Axe OG":"AutoRefractometerAxis_Left",
                        "Réfractomètre auto.:Axe OD":"AutoRefractometerAxis_Right",
                        "Verres prescrits:Sphère OG":"GlassesPrescribedSphere_Left",
                        "Verres prescrits:Sphère OD":"GlassesPrescribedSphere_Right",
                        "Verres prescrits:Cylindre OG":"GlassesPrescribedCylinder_Left",
                        "Verres prescrits:Cylindre OD":"GlassesPrescribedCylinder_Right",
                        "Verres prescrits:Axe OG":"GlassesPrescribedAxis_Left",
                        "Verres prescrits:Axe OD":"GlassesPrescribedAxis_Right",
                        "Verres prescrits:Addition OG":"GlassesPrescribedAdd_Left",
                        "Verres prescrits:Addition OD":"GlassesPrescribedAdd_Right",
                        "Verres portés:Sphère OG":"GlassesSphere_Left",
                        "Verres portés:Sphère OD":"GlassesSphere_Right",
                        "Verres portés:Cylindre OG":"GlassesCylinder_Left",
                        "Verres portés:Cylindre OD":"GlassesCylinder_Right",
                        "Verres portés:Axe OG":"GlassesAxis_Left",
                        "Verres portés:Addition OG":"GlassesAdd_Left",
                        "Verres portés:Addition OD":"GlassesAdd_Right",
                        "Verres portés:Axe OD":"GlassesAxis_Right",
                        "Acuité visuelle:Sphère OG":"VisualAcuitySphere_Left",
                        "Acuité visuelle:Sphère OD":"VisualAcuitySphere_Right",
                        "Acuité visuelle:Cylindre OG":"VisualAcuityCylinder_Left",
                        "Acuité visuelle:Cylindre OD":"VisualAcuityCylinder_Right",
                        "Acuité visuelle:Axe OG":"VisualAcuityAxis_Left",
                        "Acuité visuelle:Axe OD":"VisualAcuityAxis_Right",
                        "Biométrie:Valeur OG":"BiometryValue_Left",
                        "Biométrie:Valeur OD":"BiometryValue_Right",
                        "Kérato. en dioptries:Puissance moyenne kérato OG":"K1_Left",
                        "Kérato. en dioptries:Puissance moyenne kérato OD":"K1_Right",
                        "Kérato. en dioptries:2ème puissance kérato OG":"K2_Left",
                        "Kérato. en dioptries:2ème puissance kérato OD":"K2_Right",
                        "Tonus oculaire:Valeur OG":"Tonometry_Left",
                        "Tonus oculaire:Valeur OD":"Tonometry_Right",
                        "Tonus oculaire corr.:Valeur OG":"TonometryCorrected_Left",
                        "Tonus oculaire corr.:Valeur OD":"TonometryCorrected_Right",
                        "Tonus oculaire":"TonometryTime",
                        "Pachymétrie:Valeur OG":"Pachimetry_Left",
                        "Pachymétrie:Valeur OD":"Pachimetry_Right",
                        "Distance inter pupil":"pupilDistance",
                        "Implant D:Identif. implant":"IolID_Right",
                        "Implant D:Modèle implant":"IolModel_Right",
                        "Implant D:Famille d'implant":"IolModelFamily_Right",
                        "Implant D:Puissance implant":"IolPower_Right",
                        "Implant D:Longueur axiale":"IolAxialLength_Right",
                        "Implant D:Réfraction post-op. souhaitée":"IolDesiredRefractionPostOp_Right",
                        "Implant D:Réfraction finale":"IolFinalRefraction_Right",
                        "Implant D:Formule de calcul":"IolFormula_Right",
                        "Implant D:Kératométrie moyenne":"IolMeanK_Right",
                        "Implant D:Profondeur ch. ant. (ACD)":"IolACD_Right",
                        "Implant D:Puissance théorique":"IolTheoreticalPower_Right",
                        "Implant D:Date chirurgie prévue":"IolExpectedImplantDate_Right",
                        "Implant D:Constante A":"IOLAconst_Right",
                        "Implant OG:Identif. implant":"IolID_Left",
                        "Implant OG:Modèle implant":"IolModel_Left",
                        "Implant OG:Famille d'implant":"IolModelFamily_Left",
                        "Implant OG:Puissance implant":"IolPower_Left",
                        "Implant OG:Longueur axiale":"IolAxialLength_Left",
                        "Implant OG:Réfraction post-op. souhaitée":"IolDesiredRefractionPostOp_Left",
                        "Implant OG:Réfraction finale":"IolFinalRefraction_Left",
                        "Implant OG:Formule de calcul":"IolFormula_Left",
                        "Implant OG:Kératométrie moyenne":"IolMeanK_Left",
                        "Implant OG:Profondeur ch. ant. (ACD)":"IolACD_Left",
                        "Implant OG:Puissance théorique":"IolTheoreticalPower_Left",
                        "Implant OG:Date chirurgie prévue":"IolExpectedImplantDate_Left",
                        "Implant OG:Constante A":"IOLAconst_Left"})

        print("[Info][OplusParser] Renaming columns")
        self.data.rename(columns = nameMap, inplace = True)
        # remove all other columns
        print("[Info][OplusParser] dropping unused columns")
        colList = []
        names = list(nameMap.values())
        for k in self.data.keys():
            if np.isin(k,names)==False:
                colList.append(k)

        self.data.drop(columns = colList,inplace=True)

    @staticmethod
    def _toFloat(val):
        """
         a method to use wil
        """
        try:
            val = float(val)
        except:
            val = np.nan
        return val

    @staticmethod
    def _ParseAgeColumn(val):
        """
         method used to parse the Age column when ExamDate or birthdate are not availale
         columns in the input database

        """
        if isinstance(val,str):
            if val.find('ans')!=-1:
                val = float(val.replace('ans',''))
            elif val.find('mois')!=-1:
                val = float(val.replace('mois',''))/12
            elif val.find('semaines')!=-1:
                val = float(val.replace('semaines',''))/52
            elif val.find('semaine')!=-1:
                val = float(val.replace('semaine',''))/52
            elif val.find('jours')!=-1:
                val = float(val.replace('jours',''))/365
            else:
                val = np.nan
        else:
            return np.nan

    def Parse(self):
        self._RenameColumns()
        # Add Id column
        # emrKeys = self.data.keys()
        self.data["FirstName"] = self.data["FirstName"].str.lower()
        self.data["Surname"]   = self.data["Surname"].str.lower()
        bDate                  = list(self.data["BirthDate"].apply(self._GetDateString))
        if 'ExamDate' in self.data.columns:
            eDate            = list(self.data["ExamDate"].apply(self._GetDateString))
            age              = list(np.zeros(len(bDate)))
        if 'Age' in self.data.columns:
            age =  list(self.data['Age'].apply(self._ParseAgeColumn))

        # self.data["BirthDate"] = self.data["BirthDate"].apply(self._GetDateString)
        # self.data["ExamDate"]  = self.data["ExamDate"].apply(self._GetDateString)
        # age              = list((pd.to_datetime(self.data["ExamDate"],dayfirst=True,errors='ignore')-pd.to_datetime(self.data["BirthDate"],dayfirst=True,erros='ignore')).dt.days/365)
        # self.data["Age"] = (pd.to_datetime(self.data["ExamDate"],dayfirst=True,errors='ignore')-\
        #                     pd.to_datetime(self.data["BirthDate"],dayfirst=True,errors='ignore')).dt.days/365
        autoSphere_Left  = list(self.data["AutoRefractometerSphere_Left"].apply(self._toFloat))
        autoSphere_Right = list(self.data["AutoRefractometerSphere_Right"].apply(self._toFloat))
        autoCyl_Left     = list(self.data["AutoRefractometerCylinder_Left"].apply(self._toFloat))
        autoCyl_Right    = list(self.data["AutoRefractometerCylinder_Right"].apply(self._toFloat))
        autoAx_Left      = list(self.data["AutoRefractometerAxis_Left"].apply(self._toFloat))
        autoAx_Right     = list(self.data["AutoRefractometerAxis_Right"].apply(self._toFloat))

        # visual acuity
        vaSphere_Left    = list(self.data["VisualAcuitySphere_Left"].apply(self._toFloat))
        vaSphere_Right   = list(self.data["VisualAcuitySphere_Right"].apply(self._toFloat))
        vaCyl_Left       = list(self.data["VisualAcuityCylinder_Left"].apply(self._toFloat))
        vaCyl_Right      = list(self.data["VisualAcuityCylinder_Right"].apply(self._toFloat))
        vaAx_Left        = list(self.data["VisualAcuityAxis_Left"].apply(self._toFloat))
        vaAx_Right       = list(self.data["VisualAcuityAxis_Right"].apply(self._toFloat))

        # glasses prescribed
        gpSphere_Left    = list(self.data["GlassesPrescribedSphere_Left"].apply(self._toFloat))
        gpSphere_Right   = list(self.data["GlassesPrescribedSphere_Right"].apply(self._toFloat))
        gpCyl_Left       = list(self.data["GlassesPrescribedCylinder_Left"].apply(self._toFloat))
        gpCyl_Right      = list(self.data["GlassesPrescribedCylinder_Right"].apply(self._toFloat))
        gpAx_Left        = list(self.data["GlassesPrescribedAxis_Left"].apply(self._toFloat))
        gpAx_Right       = list(self.data["GlassesPrescribedAxis_Right"].apply(self._toFloat))

        # glasses
        gSphere_Left    = list(self.data["GlassesSphere_Left"].apply(self._toFloat))
        gSphere_Right   = list(self.data["GlassesSphere_Right"].apply(self._toFloat))
        gCyl_Left       = list(self.data["GlassesCylinder_Left"].apply(self._toFloat))
        gCyl_Right      = list(self.data["GlassesCylinder_Right"].apply(self._toFloat))
        gAx_Left        = list(self.data["GlassesAxis_Left"].apply(self._toFloat))
        gAx_Right       = list(self.data["GlassesAxis_Right"].apply(self._toFloat))

        #        pupilDistance    = list(self.data["pupilDistance"])
        if self.emrLoaded:
            for eIdx in range(len(self.data)):

                # print(f"[Info][OplusParser] parsing value {eIdx}/{len(self.data)} ({eIdx/len(self.data)*100:.2f}\%)")
                self._ProgressBar(eIdx,len(self.data))
                if 'ExamDate' in self.data.columns:
                    age[eIdx]   = self._GetAge(bDate[eIdx], eDate[eIdx])

                # select a string for parsing
                sFlagLeft  = autoSphere_Left[eIdx].find('#')!=-1 if  isinstance(autoSphere_Left[eIdx],str) else False
                sFlagRight = autoSphere_Right[eIdx].find("#")!=-1 if isinstance(autoSphere_Right[eIdx], str) else False
                if sFlagLeft or sFlagRight:
                    sph,cyl,ax,pupDist = self._ParseSphCylAxFromString(autoSphere_Left[eIdx],autoSphere_Right[eIdx])
                    autoSphere_Right[eIdx] = float(sph[0])
                    autoSphere_Left[eIdx]  = float(sph[1])
                    autoCyl_Right[eIdx]    = float(cyl[0])
                    autoCyl_Left[eIdx]     = float(cyl[1])
                    autoAx_Right[eIdx]     = float(ax[0])
                    autoAx_Left[eIdx]      = float(ax[1])


                # parse visual acuity
                sFlagLeft  = vaSphere_Left[eIdx][0] == '#' if  isinstance(vaSphere_Left[eIdx],str) else False
                sFlagRight = vaSphere_Right[eIdx][0] == '#' if isinstance(vaSphere_Right[eIdx],str) else False
                if sFlagLeft or sFlagRight:
                    sph, cyl,ax, pupDist = self._ParseSphCylAxFromString(vaSphere_Left[eIdx],vaSphere_Right[eIdx])
                    vaSphere_Right[eIdx] = float(sph[0])
                    vaSphere_Left[eIdx]  = float(sph[1])
                    vaCyl_Right[eIdx]    = float(cyl[0])
                    vaCyl_Left[eIdx]     = float(cyl[1])
                    vaAx_Right[eIdx]     = float(ax[0])
                    vaAx_Left[eIdx]      = float(ax[1])

                # parse glasses prescribed
                sFlagLeft  = gpSphere_Left[eIdx][0] == '#' if  gpSphere_Left[eIdx].__class__ == str else False
                sFlagRight = gpSphere_Right[eIdx][0] == '#' if gpSphere_Right[eIdx].__class__ == str else False
                if sFlagLeft or sFlagRight:
                    sph,cyl,ax,pupDist = self._ParseSphCylAxFromString(gpSphere_Left[eIdx],gpSphere_Right[eIdx])
                    gpSphere_Right[eIdx] = float(sph[0])
                    gpSphere_Left[eIdx]  = float(sph[1])
                    gpCyl_Right[eIdx]    = float(cyl[0])
                    gpCyl_Left[eIdx]     = float(cyl[1])
                    gpAx_Right[eIdx]     = float(ax[0])
                    gpAx_Left[eIdx]      = float(ax[1])

                # parse glasses
                sFlagLeft  = gSphere_Left[eIdx][0] == '#' if  gSphere_Left[eIdx].__class__ == str else False
                sFlagRight = gSphere_Right[eIdx][0] == '#' if gSphere_Right[eIdx].__class__ == str else False
                if sFlagLeft or sFlagRight:
                    sph,cyl,ax,pupDist = self._ParseSphCylAxFromString(gSphere_Left[eIdx],gSphere_Right[eIdx])
                    gSphere_Right[eIdx] = float(sph[0])
                    gSphere_Left[eIdx]  = float(sph[1])
                    gCyl_Right[eIdx]    = float(cyl[0])
                    gCyl_Left[eIdx]     = float(cyl[1])
                    gAx_Right[eIdx]     = float(ax[0])
                    gAx_Left[eIdx]      = float(ax[1])
                if gSphere_Right[eIdx].__class__==str:
                    print('[Warning][OplusParser] Still a string')


            # if fName:
            # self.data["FirstName"] = fName
            # if sName:
            # self.data["Surname"]   = sName
            # if bDate:
            self.data["BirthDate"] = bDate
            # if eDate:
            if "ExamDate" in self.data.columns:
                self.data["ExamDate"]  = eDate

            # if age:
            self.data["Age"]       = age

            # Assign autorefractometer (objective)
            self.data["AutoRefractometerSphere_Left"]    = autoSphere_Left
            self.data["AutoRefractometerSphere_Right"]   = autoSphere_Right
            self.data["AutoRefractometerCylinder_Left"]  = -np.abs(autoCyl_Left)
            self.data["AutoRefractometerCylinder_Right"] = -np.abs(autoCyl_Right)
            self.data["AutoRefractometerAxis_Left"]      = autoAx_Left
            self.data["AutoRefractometerAxis_Right"]     = autoAx_Right
            for eyIdx in ['_Right','_Left']:
                badInds = self.data[f'AutoRefractometerSphere{eyIdx}'].notna() & self.data[f'AutoRefractometerCylinder{eyIdx}'].isna()
                self.data.loc[badInds,f'AutoRefractometerCylinder{eyIdx}'] = 0
                self.data.loc[badInds,f'AutoRefractometersAxis{eyIdx}']    = 0

            # Assign subjective refraction
            self.data["VisualAcuitySphere_Left"]         = vaSphere_Left
            self.data["VisualAcuitySphere_Right"]        = vaSphere_Right
            self.data["VisualAcuityCylinder_Left"]       = -np.abs(vaCyl_Left)
            self.data["VisualAcuityCylinder_Right"]      = -np.abs(vaCyl_Right)
            self.data["VisualAcuityAxis_Left"]           = vaAx_Left
            self.data["VisualAcuityAxis_Right"]          = vaAx_Right
            for eyIdx in ['_Right','_Left']:
                badInds = self.data[f'VisualAcuitySphere{eyIdx}'].notna() & self.data[f'VisualAcuityCylinder{eyIdx}'].isna()
                self.data.loc[badInds,f'VisualAcuityCylinder{eyIdx}'] = 0
                self.data.loc[badInds,f'VisualAcuityAxis{eyIdx}']     = 0

            # Assign prescribed glasses measurements
            self.data["GlassesPrescribedSphere_Left"]    = gpSphere_Left
            self.data["GlassesPrescribedSphere_Right"]   = gpSphere_Right
            self.data["GlassesPrescribedCylinder_Left"]  = -np.abs(gpCyl_Left)
            self.data["GlassesPrescribedCylinder_Right"] = -np.abs(gpCyl_Right)
            self.data["GlassesPrescribedAxis_Left"]      = gpAx_Left
            self.data["GlassesPrescribedAxis_Right"]     = gpAx_Right
            for eyIdx in ['_Right','_Left']:
                badInds = self.data[f'GlassesPrescribedSphere{eyIdx}'].notna() & self.data[f'GlassesPrescribedCylinder{eyIdx}'].isna()
                self.data.loc[badInds,f'GlassesPrescribedCylinder{eyIdx}'] = 0
                self.data.loc[badInds,f'GlassesPrescribedAxis{eyIdx}']     = 0

            # assign glasses measurements
            self.data["GlassesSphere_Left"]    = gSphere_Left
            self.data["GlassesSphere_Right"]   = gSphere_Right
            self.data["GlassesCylinder_Left"]  = -np.abs(gCyl_Left)
            self.data["GlassesCylinder_Right"] = -np.abs(gCyl_Right)
            self.data["GlassesAxis_Left"]      = gAx_Left
            self.data["GlassesAxis_Right"]     = gAx_Right
            for eyIdx in ['_Right','_Left']:
                badInds = self.data[f'GlassesSphere{eyIdx}'].notna() & self.data[f'GlassesCylinder{eyIdx}'].isna()
                self.data.loc[badInds,f'GlassesCylinder{eyIdx}'] = 0
                self.data.loc[badInds,f'GlassesAxis{eyIdx}']     = 0

        return self.data

    @ staticmethod
    def _ProgressBar(cur_val, max_val,perc_interval=5,fill='#'):
        """
         Draw a progress bar

         Parameters
         -----------
         cur_val (int) :
            current iteration
         max_val (int) :
            maximal number of interation in the loop
         perc_interval (int, float) : default = 5
           jumps of the progress bar, must be < 100
         fill (str) : default = '#
           graphics to diplay with each step of the progress bar


        """
        # Check validity of input variables
        if not isinstance(fill,str):
            fill = f'{fill}' # make string
        if not isinstance(perc_interval,(int,float)):
            raise ValueError(f'perc_interval must be numeric. Got {perc_interval.__class__}')
        if perc_interval>100:
            perc_interval = 10 # resort to default
        if perc_interval<=0:
            perc_interval = 10 # resort to default
        if not isinstance(max_val,(int,float)):
            raise ValueError(f'max_val must be numeric. got {max_val.__class__} ')
        if max_val<=0:
            raise ValueError(f'max_val must be positive integer. Got {max_val}')

        perc = fill*round((cur_val/max_val)*100/perc_interval)*perc_interval+'->'

        print(f'\r {perc} ({100*cur_val/max_val:.2f}\%)', end='\r')

    @staticmethod
    def _ToLowerCase(val):
        # Transform a value to lowercase, if not string return np.nan
        s = val.lower() if val.__class__ == str else val
        return s

    @staticmethod
    def _GetDateString(d, yearFirst=False):
        """
         Return a string representing date. if yearFirst=False dd/mm/yyyy or yyyy/mm/dd if yearFirst=True
         defalut format is dd/mm/yyyy
         Parameters:
         ---------
         d, str,
           data string
         Output:
         --------
         formated date string
        """
        if isinstance(d,str):
            if len(d.split("/"))==3:
                return d
            else:
                try:
                    s = dateutil.parser.parse(d,dayfirst=True)
                    if s.year>datetime.date.today().year: # for cases e.g. 01/01/20
                            s =s.replace(year=s.year-100)

                    if yearFirst:
                        return f"{s.year}/{s.month:02d}/{s.day:02d}"
                    else:
                        return f"{s.day:02d}/{s.month:02d}/{s.year}"
                except:
                    return np.nan

    @staticmethod
    def _GetAge(birthDate,examDate):
        # compute age at examination time by subtracting birthdate from exam date
        # birthDate -str
        # examDate- str
        try:
            t = datetime.datetime.today()
            a = dateutil.parser.parse(birthDate)
            b = dateutil.parser.parse(examDate)
            if a>t or b>t: # check for correct registration of dates
                return np.nan
            else:
                age = (b-a).days/365
                if age<0:
                    age += 100
                return age
        except:
            return np.nan

    @staticmethod
    def _StringReplace(string, d):
        # Replace all appearances of symbols in dictionary d in string
        # d is a dictionary of format {oldVal1:newVal1,oldVal2:newVal2,...}

        if string.__class__== str and d.__class__ == dict:
            k = d.keys()
            for kIdx in k:
                string = string.replace(kIdx, d[kIdx])
        return string

    def _ParseSphCylAxFromString(self,fVal_Left,fVal_Right):
        """
         Correct non-uniform registration of Sphere Cylinder and Axis in the Oplus database
         by different practitioners.
         A summary is sometimes written in the field RefractometreAutoSphereOD in the
         format #+S(C)A°/S(C)A°* pupDist=70.0 for OS/OD
         This record needs to be broken into its elements and inserted in the dedicated fields for that record
         sometimes a writing as the following exist: #PL(-3.25)13° / +0.75(-3.00)176°* pupDist=60.0,
         The PL signifies Plano, that is 0 sphere (neither near nor far sighted), thus, PL should be replaced with S=0
        """

        # Pre-allocation
        fVal    = np.nan
        pupDist = np.nan
        sph     = np.ones(2)*np.nan
        cyl     = np.ones(2)*np.nan
        ax      = np.ones(2)*np.nan
        lFlag   = isinstance(fVal_Left,str)
        rFlag   = isinstance(fVal_Right,str)

        if lFlag and rFlag: # both are strings
            # check which one contains the # symbol
            if fVal_Left[0].find('#')!=-1:
                fVal = fVal_Left
            elif fVal_Right[0].find('#')!=-1:
                fVal = fVal_Right
        elif lFlag and not rFlag:  # left is string right is not
            if fVal_Left[0].find('#')!=-1:
                fVal = fVal_Left
        elif not lFlag and rFlag:  # right is string left is not
            if fVal_Right[0].find('#')!=-1:
                fVal = fVal_Right

        if isinstance(fVal,str): # if the symbol is found in either the left or right value
            if fVal[0] == '#':
                # Strip the string of the indication of pupDist
                fVal = self._StringReplace(fVal,
                                            {"#":"",
                                            "+":"",
                                            "*":"",
                                            " ":"",
                                            "°":"",
                                            "PL":"0.00",
                                            "SKIACO":"",
                                            "SKAICO":"",
                                            " ":"",
                                            "lentilles":""})  # remove special symbols

                fVal = fVal.split("pupDist=")
                if len(fVal) == 2: # if pupDist= is found
                    r    = re.findall(r'\d+.\d+', fVal[1])
                    try:
                        pupDist   = np.float(r[0]) if len(r) > 0 else np.nan  # take the first one found as the pupDist
                    except:
                        pupDist = np.nan
                    fVal = fVal[0]
                else: # if pupDist= is not found
                    pupDist   = np.nan
                    fVal = fVal[0]
                # Break the string into left and right eyes and pupDist
                lr = fVal.split("/")

                fR = lr[0] if len(lr)==2 else np.nan #only right eye recorded
                fL = lr[1] if len(lr)==2 else np.nan

                nextInds = 0 # start with left
                for fStr in (fL, fR):
                    # try:
                    if isinstance(fStr,str):
                        pattern1 = re.findall(r'[+-]?\d+\.\d+\([+-]?\d+\.\d+\)\d+',fStr)
                        if len(pattern1)==1:
                            # extract sph cyl ax
                            fStr = pattern1[0]
                            cIndStart     = fStr.index("(")
                            cIndEnd       = fStr.index(")")
                            sph[nextInds] = float(fStr[:cIndStart])
                            cyl[nextInds] = -np.abs(np.float(fStr[cIndStart+1:cIndEnd]))
                            ax[nextInds]  = np.float(fStr[cIndEnd+1:])
                        else:
                            pattern2 = re.findall(r'[+-]?\d+\.d+',fStr) # sphere only
                            if len(pattern2)==1:
                                sph[nextInds] = float(pattern2[0])
                                cyl[nextInds] = 0
                                ax[nextInds]  = 0
                    nextInds += 1
        return sph, cyl, ax, pupDist