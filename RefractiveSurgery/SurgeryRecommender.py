import math
import pandas as pd

class Recommender:
    """
     Myopic or Hyperopic refractive surgery recommender.
     For patients with spherical equivalent in [0.25, -6D] we activate the Myopic recommender
     For patients with spherical equivalent <-6, we activate the hyperopic recommender
    """
    def __init__(self):
        # mapping of surgery types into integer (string) representation
        self._ver = 0.1
        self._surgeries = {1:'PRK MMC WL Allegreto',
                           2:'PRK MMC Teneo HD',
                           3:'PRK MMC Teneo Proscan',
                           4:'PRK MMC Teneo Transepith',
                           5:'Smile Femto Visumax',
                           6:'FemtoLasik Femto + WL Allegretto',
                           7:'UltraLasik Femto + WFG Teneo HD',
                           8:'Bioptic ICL + FemtoLasik',
                           9:'ICL Visian Spheric',
                           10:'ICL Visian Toric',
                           11:'Phaco + IOL/Prelex Multifocal/Monovision',
                           12:'Phaco + Customized IOL',
                           13:'FemtoLasik Femto + Teneo Proscan',
                           14:'PresbyLasik Femto + WL -1.25 Q=-0.8',
                           15:'PresbyLasik Femto + Teneo Supracor',
                           16:'Cataract'
                           }

    def TestAllRecommendations(self,res:pd.DataFrame,reqCorD:tuple,opticalZoneDiameter=(6.5,6.5),pvd=(False,False),
                        priorLasik=(False,False),dominantEye='right',flapThickness=(100,100),
                        topoPattern=('normal','normal'),
                        epithelial_pattern=('normal','normal'))->dict:
        """
         Test function to run inclusion criteria for all surguries possible and return a list of
         recommended surgeries. Add priority rank to surgery types on the list given

         Parameters
         -----------
         res : DatraFrame
          results structure (see vx120Transformer)

        """
        # [Experimental, INCOMPLETE]
        # compute all needed parameters
        bank = {'PRK MMC WL Alegrato': {'cyl':(-0.75,0),
                                        'hoa':(0,0.6),
                                        'rsb':(400, math.inf),
                                        'se':(-2.0,0.25)},
                'PRK MMC Teneo HD': {'or':{'cyl':(-math.inf, -0.75),'hoa':(0.6,math.inf)},
                                    'rsb':(400,math.inf),'se':(-2,0.25)}
                }

        eye   = ['_Right','_Left']
        for eIdx in range(2):
            ind         = res.index[0]
            se          = res.loc[ind,f'SphericalEquivalent_3{eye[eIdx]}']
            meanK       = res.loc[ind,f'Topo_Sim_K_Avg{eye[eIdx]}']
            pachy       = res.loc[ind,f'Pachy_MEASURE_Thickness{eye[eIdx]}']
            kpi         = res.loc[ind,f'Topo_KERATOCONUS_Kpi{eye[eIdx]}']
            cyl         = res.loc[ind,f'WF_CYLINDER_R_3{eye[eIdx]}']
            hoa         = res.loc[ind,f'WF_ZERNIKE_3_HOA{eye[eIdx]}']
            age         = res.loc[ind,'Age']
            acd         = res.loc[ind,f'Pachy_MEASURE_Acd{eye[eIdx]}']
            axialLength = res.loc[ind,f'AxialLength_Avg{eye[eIdx]}']
            angR        = res.loc[ind,f'Pachy_MEASURE_IridoAngleR{eye[eIdx]}']*math.pi/180
            angL        = res.loc[ind,f'Pachy_MEASURE_IridoAngleL{eye[eIdx]}']*math.pi/180
            meanIridoAngle = math.atan2((math.sin(angR)+math.sin(angL))/2,(math.cos(angR)+math.cos(angL))/2)*180/math.pi
            rsb = self._ComputeRSB(pachy,reqCorD,opticalZoneDiameter, flapThickness)


        return None

    def RunFromParsedData(self,vx_data:pd.DataFrame,pred_ref:pd.DataFrame,
                        target_refraction:tuple,pvd=(False,False),
                        priorLasik=(False,False),
                        dominantEye=None,
                        flapThickness=(100,100),
                        topoPattern=('normal','normal'),
                        epithelial_pattern=('normal','normal'),
                        diopter_interval = 0.25,
                        vertex_distance = 0.012)->dict:
        """
            A service function to prepare input from parsed vx120 and oct data

            Parameters
            ----------
            vx_data : DataFrame
                parsed and transformed vx120 data (see vx120Transformer) and OCT measurements (see Revo parser)
                res must include keys with _Left and _Right suffix for objective measurements
                indicating left and right eye data
            pred_ref : DataFrame
              predicted refraction, dataframe resuling from subjective refractino Predictor
              All fields have initial Predicted_ and suffix either _Left or _right
              to indicate the eye
            target_refraction : tuple float
                planed target refraction (sphere only) D, target refraction - subjective refraction (sphere only)
            opticalZonediameter : tuple float
                diameter of optical zone for ablation (mm)
            dominantEye : str
                dominant eye (right/left/None)
                if dominantEye  is set to None, the dominant eye is chosen to be the least astigmatic one
                with the lowest spherical equivalent (best refraction)
            flapThickness : tuple float,
                flap thickness for Lasik
            topoPattern : tuple str
                pattern in the corneal topography
            vertex_distance : float, default = 0.012
                vertex distance (m) to trnalate the subjective sphere to the corneal plane
            diopter_interval : float, default=0.25
                round sph, cyl to nearest diopter_interval (D)

            Output
            ---------
            recommendation, list of strings
             recommended surgery for right and left eye

        """

        recom = {'Right':{},'Left':{}}
        eye   = ['Right','Left']

        # determine dominant eye if not suppliedr
        if dominantEye is None:
            dominantEye = self._GetDominantEye((vx_data.iloc[0]['SphericalEquivalent_3_Right'],vx_data.iloc[0]['SphericalEquivalent_3_Left']),
                                            (vx_data.iloc[0]['WF_CYLINDER_R_3_Right'],vx_data.iloc[0]['WF_CYLINDER_R_3_Left']))
            print(f'[Info][SurgeryRecommender] dominant eye not supplied. Setting dominant eye to {dominantEye}')

        for eIdx in range(2):
            ind         = vx_data.index[0]
            se          = vx_data.loc[ind,f'SphericalEquivalent_3_{eye[eIdx]}']
            meanK       = vx_data.loc[ind,f'Topo_Sim_K_Avg_{eye[eIdx]}']
            pachy       = vx_data.loc[ind,f'Pachy_MEASURE_Thickness_{eye[eIdx]}']
            kpi         = vx_data.loc[ind,f'Topo_KERATOCONUS_Kpi_{eye[eIdx]}']
            cyl         = vx_data.loc[ind,f'WF_CYLINDER_R_3_{eye[eIdx]}']
            hoa         = vx_data.loc[ind,f'WF_ZERNIKE_3_HOA_{eye[eIdx]}']
            age         = vx_data.loc[ind,'Age']
            acd         = vx_data.loc[ind,f'Pachy_MEASURE_Acd_{eye[eIdx]}']
            axialLength = vx_data.loc[ind,f'AxialLength_Avg_{eye[eIdx]}']
            angR        = vx_data.loc[ind,f'Pachy_MEASURE_IridoAngleR_{eye[eIdx]}']*math.pi/180
            angL        = vx_data.loc[ind,f'Pachy_MEASURE_IridoAngleL_{eye[eIdx]}']*math.pi/180
            pupilDiam   = vx_data.loc[ind,f'WF_RT_Fv_Photo_PupilDiameter_{eye[eIdx]}']
            opticalZoneDiameter = 6.5 if pupilDiam<6.5 else pupilDiam
            meanIridoAngle = math.atan2((math.sin(angR)+math.sin(angL))/2,(math.cos(angR)+math.cos(angL))/2)*180/math.pi
            # predict for each eye (place right before left eye in the list)
            # use the predicted subjective sphere translated into the corneal plane
            sph_cornea = pred_ref[f'Predicted_Sphere_{eye[eIdx]}']/(1-vertex_distance*pred_ref[f'Predicted_Sphere_{eye[eIdx]}'])
            sph_cornea = (sph_cornea/diopter_interval).round()*diopter_interval
            reqCorD    = (target_refraction[eIdx]-sph_cornea).loc[ind]




            d =  self.Run(se,meanK,pachy,kpi,cyl,hoa,age,
                            acd,axialLength,meanIridoAngle,reqCorD,
                            priorLasik[eIdx],pvd[eIdx],opticalZoneDiameter,
                            dominantEye,flapThickness[eIdx],topoPattern=topoPattern[eIdx],epithelial_pattern=epithelial_pattern[eIdx])
            for dIdx in d.keys():
                recom[eye[eIdx]][dIdx] = d[dIdx]

        return recom

    @staticmethod
    def _GetDominantEye(se:tuple,cyl:tuple)->str:
        """
         Set dominant eye according to the eye with the best refraction

        Args:
        ------
            se (tuple(float)):
              spherical equivalent for (right, left) eye
            cyl (tuple(float,flaot)):
              cylinder power for (right,left) eye

        Returns:
        ---------
         dominantEye (str):
          the dominant eye (right/left)

        """
        sph = (se[0]-cyl[0]/2, se[1]-cyl[1]/2)
        if abs(se[0])<abs(se[1]):
            if abs(sph[0])<abs(sph[1]):
                dominantEye = 'right'
            else:
                dominantEye = 'left'
        elif se[0]==se[1]:
            # take the least astigmatic eye
            if abs(cyl[0])<=abs(cyl[1]):
                dominantEye = 'right'
            else:
                dominantEye = 'left'
        else:
            if abs(sph[0])>abs(sph[1]):
                dominantEye = 'left'
            else:
                dominantEye  = 'right'

        return dominantEye

    def _AgeRule(self,sType,age:float,dominantEye:str)->str:
        """
         sType: str
          surgery type, either myopic or hyporopic
         age : flaot
          age of patient, positive float
         cominantEye: str
          either left or right

        """
        if not isinstance(age,(float,int)):
            raise ValueError(f'age must be numeric and positive, got {age.__class__}')
        else:
            if age<=0:
                raise ValueError(f'age must be striclty positive, got age={age}')

        if not isinstance(dominantEye,str):
            raise ValueError(f'dominantEye must be a string Left or Right, got {dominantEye.__clas__}')

        decision = None
        if dominantEye.lower()=='right':
             nonDominantEye = 'left'
        elif dominantEye.lower()=='left':
            nonDominantEye = 'right'
        else:
            raise ValueError(f'dominantEye must be right or left, got {dominantEye}')

        if not isinstance(sType,str):
            raise ValueError(f'sType must be a string of either myopic or hypoeropic. got {sType}')

        if sType.lower()=='myopic':
            cor = 'Under'
        elif sType.lower()=='hyperopic':
            cor = 'Vver'
        else:
            raise ValueError(f'sTytpe myst be eiter myopic or hyperopic. got {sType}')

        if age<=18:
            decision ='Not eligable'
        elif age>18 and age<=23:
            decision = 'Verify 2-year stability: refraction and topo. {cor} correct -0.25D'
        elif age>23 and age<=38:
            if age<=25:
                decision = 'Standard protocol. Verify 2-year stability: refraction and topo. {cor} correct -0.25D'
            else:
                decision = 'Standard protocol'
        elif age>38 and age<=40:
            decision = f'{cor}-correct {nonDominantEye} eye 0.50 D'
        elif age>40 and age<=45:
            decision = f'{cor}-correct {nonDominantEye} eye 1.00 D'
        elif age>45 and age<=50:
            decision = f'{cor}-correct {nonDominantEye} eye 1.50 D'
        elif age>50 and age<=55:
            decision = f'{cor}-correct {nonDominantEye} eye 1.75 D'
        elif age>55 and age<=60:
            decision = f'{cor}-correct {nonDominantEye} eye 2.25 D'
        elif age>60 and (age<=65):
            decision = f'Prefer cataract surgery'
        return decision

    def _ComputeRSB(self,pachy,reqCorD,opticZoneDiameter,flapThickness)->float:
        """
            Compute the ablation depth based on the required correctoin, then obtain the residual stromal bed after surgery
            The Munnerlyn method is used to compute the ablation depth.

            Flap thickness should be around: 100-120µm for LASIK, 140-160µm for Smile or 60µm for PRK
            Optical zone should depend on the pupil size and the type of laser

            Parameters:
            ------------
            reqCorD : float
               required correction (D)
            pachy : float
               mean pachimetry (mu m)
            opticZoneDiameter : float, default=6.5 (mm)
               optic zone diameter for surgery
            flapThickness : float
             flap thickness for Lasik (µm),
             for PRK set to 0

             Output:
             -------
              rsb : float
               residual stromal bed (µm)
        """
        ablation_depth = abs(reqCorD)*(opticZoneDiameter**2)/3 # mu m by Munnerlyn formula
        rsb = pachy-flapThickness-ablation_depth
        if rsb<0:
            raise ValueError(f'Calculated RSB resulted in negative value ={rsb}. Please check units of input parameters')
        return rsb

    def _PredictKChangePostOp(self,se:float, meanK:float,correction:float)->object:
        """
         For myopia treatment, the cornea flattens (K reduces) with each diopter corrected by 0.8
         For hyperopia, the cornea steepens (K increases) with each diopter corrected by 1D.

         Parameters:
         -----------
         se, float
          spherical equivalent (D)
         meanK, float
          mean keratometry (D)
         correction, float
          the amount of correction (D) planned for surgery

         Returns:
         --------
          keratometry post op., in D units

        """
        if se<=0.25 and se!=None:
            return meanK-0.8*correction
        elif se>0.25 and se!=None:
            return meanK-correction
        else:
            return None

    def _ComputePTA(self,ablationDepth:float,flapThickness:float,cct)->float:
        """
         Compute the Precetage of Tissue Altered after Lasik refractive surgery.
         This method was suggested by Santhiago et al. 2014.
         A PTA value greater than 40% is associated with higher risk to
         develop ecstesia post Lasik.

         Parameters
         -----------
         ablationDepth : float
           ablation depth according to Lasik (mm)
         flapThickness : float
           flap thickness (mm)
         cct : float
          central corneal thicknes pre-op (mm)

         Output
         ----------
         pta, float
          the predicted percent of tissue altered
        """

        return (flapThickness+ablationDepth)/cct

    def _ComputeRetinalDetachmentRisk(self,priorPVD,taboccoSmoke,visualAcuity,gender):
        """
           [UNFINISHED]
           Compute the risk of retinal detachment based on the BElfast risk score
           to distinguish it from PVD
        """

        return None

    def Run(self,se,meanK,pachy,kpi,cyl,hoa,age,acd,axialLength,meanIridoAngle,
            reqCorD,priorLasik=False,pvd=False,opticalZoneDiameter=6.5,dominantEye='right',
            flapThickness=100,topoPattern='normal',epithelial_pattern='normal'):
        """
            Recommend a refractive surgery type based on objective measurements

            Parametes
            -------
            se : float
                spherical equivalent for meso (D)
            meanK : float
                mean keratometry (D)
            pachy : float
                pchimetry (mu m)
            kpi : int
                keratoconus index
            cyl : float
                cylinder for Meso (D)
            hoa : float
                high order aberrations from Topo at Meso pupil
            age : float
                age at examination time (years)
            axialLength : float
                axial length (mm)
            acd : float
                anterior chamber depth
            meanIridoAngle : float
                average irido angle (left, right, of the same eye)
            reqCorD : float
                required correction (D)
            priorLasik : bool, default=False
                prior lasik operations (true/false)
            pvd : bool, default = False
                partial vitreous detachment
            topoPattern : str, default = 'normal'
                prr-op condition based on topo map
                either ffkc, inferior_steepening, sra, abt, normal, sbt
                with the abbrreviations:
                 ffkc- forme fruste keratoconus
                 sra - skewed radial axis
                 sbt - symmetric bowtie
                 abt - asymmetric bowtie

            Output
            --------
            decision : str
                recommendation for a surgery,
                if no recommendation can be given, decision = None
        """

        # Chack validity of input variables
        params = locals().copy()
        params.pop('self')
        self._VerifyClassInputVariables(params)
        # Define output dictionary
        serg = {}
        for k in list(self._surgeries.values()):
            serg[k] = None
        res = {'Type':None,
               'Decision':serg,
               'EctasiaRiskPTA':None,
               'EctasiaRiskRendleman':None,
               'RetinalDetachmentRisk':None,
               'CompletePVD':None,
               'AgeRule':None,
               'PTA':None,
               'RSB':None,
               'AblationDepth':None,
               'PredictedKPostOp':None,
               'params':params}

        if se is not None:
            # reqCorD = target_refraction-subj_sph # correction sphere only

            if se<=0.25:
                decision,acceptable = self._MyopicRecommender(se,meanK,pachy,kpi,cyl,hoa,age,
                                                    acd,axialLength,meanIridoAngle,
                                                    reqCorD,priorLasik,pvd,
                                                    opticalZoneDiameter,flapThickness,
                                                    epithelial_pattern)
                sType = 'myopic'
            elif se>0.25:
                decision, acceptable = self._HyperopicRecommender(se,cyl,meanK,pachy,kpi,age,acd,meanIridoAngle,priorLasik,pvd)
                sType = 'hyperopic'
            else:
                decision   = None
                acceptable = []
                sType      = None

            if decision is not None:

                res['Type']                 = sType
                ablationDepth               = self._ablation_depth(reqCorD,opticalZoneDiameter)
                res['AblationDepth']        = ablationDepth
                res['EctasiaRiskPTA'],pta   = self._EctasiaRisk_PTA(pachy,flapThickness,ablationDepth)
                res['EctasiaRiskRendleman'] = self._EctasiaRisk_Rendleman(age,pachy,se,opticalZoneDiameter,
                                                                                flapThickness,reqCorD,
                                                                                topoPattern=topoPattern)
                # estimate ectasia risk for patients with normal topo map
                res['AgeRule']          = self._AgeRule(sType,age,dominantEye)
                res['PTA']              = pta
                res['PredictedKChange'] = self._PredictKChangePostOp(se,meanK,reqCorD)
                res['Decision'][decision] = 'Recommended'
            for aIdx in acceptable:
                res['Decision'][aIdx] = 'Acceptable'

        return res

    def _EctasiaRisk_PTA(self,pachy:float,flapThickness:float,ablationDepth:float)->tuple:
        """
         Estimate the risk of ectasia post Lasik for patients with normal topo map,
         based on the percent of tissue alterd.

         Parameters
         ----------
         pachy : float
          pachymetry (mu m)
         flapThickness : float
          Lasik flap thickness (mm)
         ablationDepth : float
          ablation depth (mm)

         Output
         ---------
         risk : str
          either Low, Medium, or High
         pta : float
            predicted percent of tissure altered post Lasik
        """

        pta  = self._ComputePTA(ablationDepth,flapThickness,pachy)

        if pta<0.4:
            return 'Low',pta
        elif (pta>=0.4) and (pta<0.7):
            return 'Medium',pta
        elif pta>0.7:
            return 'High',pta
        else:
            return None,pta

    @staticmethod
    def _ablation_depth(reqCorD,ozd):
        """
         Predict ablation depth

         Parameters
         -------
         reqCorD : float
           required correction (D)
         ozd : float
           optical zone diameter (mm)

         Output
         ------
         ablation depth (micron), float

        """
        return abs(reqCorD)*(ozd**2)/3

    def _EctasiaRisk_Rendleman(self,age,pachy,se,opticalZoneDiameter,flapThickness,reqCorD,topoPattern='normal'):
        """
         Compute the Rendleman ectasia risk
         Parameters:
         -----------
         age, float
          age in years
         pachy, float
          central pachymetry (mu m)
         se, float
          spherical equivalent pre-op (D)
         opticalZoneDiameter, float
          optical zone diameter for ablation (mm)
         flap thicknes, float
          flap thickness for Lasik (mm)
         reqCorD, float
          required correction of the Lasik (D)
         topoPattern, str, default = 'normal'
          propo condition based on topo map
          either ffkc, inferior_steepening, sra, abt, normal, sbt
          with the abbrreviations:
              ffkc- forme fruste keratoconus
              sra - skewed radial axis
              sbt - symmetric bowtie
              abt = asymmetric bowtie

         Output:
         --------
         risk, str
          low: score=0-2
          medium: score=3
          high: score>3
        """
        score = 0
        if (age>=18) and (age<22):
            score+=3
        elif (age>=22) and (age<26):
            score+=2
        elif (age>=26) and (age<30):
            score+=1

        if (pachy<451):
            score+=4
        elif (pachy>=451) and (pachy<481):
            score+=3
        elif (pachy>=481) and (pachy<=510):
            score+=2

        if (se<-14):
            score+=4
        elif (se<-12) and (se>=-14):
            score+=3
        elif (se<=-10) and (se>-12):
            score+=2
        elif (se<=-8) and (se>-10):
            score+=1
        # compute rsb
        rsb = self._ComputeRSB(pachy,reqCorD,opticalZoneDiameter,flapThickness)
        if (rsb<=240):
            score+=4
        elif (rsb>240) and (rsb<260):
            score+=3
        elif (rsb>=260) and (rsb<280):
            score+=2
        elif (rsb>=280) and (rsb<300):
            score+=1

        if topoPattern.lower()=='ffkc':
            score+=4
        elif topoPattern.lower() in ['inferior_steepening','sra']:
            score+=3
        elif topoPattern.lower()=='abt':
            score+=1
        elif topoPattern.lower() in ['normal','sbt']:
            score+=0
        else:
            raise ValueError(f'topoPattern is not any of  normal,abt,sbt,sra,or ffkc. Got {topoPattern}')

        if score<3:
            return 'Low'
        elif score==3:
            return 'Medium'
        elif score>3:
            return 'High'

    def _MyopicRecommender(self,sphericalEq,meanK,pachy,kpi,cyl,hoa,age,acd,axialLength,meanIridoAngle,reqCorD,priorLasik,pvd,opticalZoneDiameter,flapThickness,epithelial_pattern):
        """
            Decision tree for negative spherical equivalent
            Decision is given in the form of a dictionary key
            see mapping in self._surgeries
            acceptable surgeries are those which are
            possible
        """
        params = locals().copy()
        params.pop('self')
        decision   = None
        acceptable = []

        flag = self._VerifyFunctionInputVariables(params)
        if flag:
            if (sphericalEq<=0.25) and (sphericalEq>=-1.5):
                decision,acceptable = self._MyopicLow(sphericalEq,cyl,hoa,pachy,reqCorD,opticalZoneDiameter,flapThickness)
            elif (sphericalEq<-1.5) and (sphericalEq>=-8):
                decision,acceptable = self._MyopicMedium(sphericalEq,meanK,pachy,hoa,kpi,cyl,
                                            meanIridoAngle,age,acd,axialLength,
                                            reqCorD,priorLasik,pvd,
                                            opticalZoneDiameter,flapThickness,
                                            epithelial_pattern)

            elif (sphericalEq<-6) and (sphericalEq>=-24):
                decision,acceptable = self._MyopicHigh(sphericalEq,cyl,hoa,axialLength,
                                            acd,meanIridoAngle,age,
                                            priorLasik=priorLasik,
                                            pvd=pvd)
        return decision,acceptable

    def _MyopicLow(self,se,cyl,hoa,pachy,reqCorD,opticalZoneDiameter,flapThickness):
        """
            Decision  for spherical eq. between 0 to -2

            Parameters:
            -----------
            se, float
             spherical equivalent (diopter)
            cyl, float
             cylinder (diopter), must be negative
            hoa, float
             high order aberrations from optical wavefront
            reqCorD, float
             required correction in D
            opticalZoneDiameter, float
             optical zone diameter for ablation (mm)
            flapThickness, float, default = 160 mu m
             the flap thickness for Lasik (mu m)

             Returns:
             ----------
             decision: str
              a decision based on the keys in self._surgeries
             acceptable: list (str)
              a list of acceptable surgeries from the list in
              self._surgeries, which are also an acceptable procedure

        """
        decision   = None
        acceptable = []
        rsb        = self._ComputeRSB(pachy,reqCorD,opticalZoneDiameter,flapThickness)
        if (se<=-0.25) and (se>-6) and (rsb>=400):
            if (cyl>=-0.75) and (hoa<=0.6):
                decision  = self._surgeries[1]        # "PRK MMC WL Allegretto"
                acceptable.append(self._surgeries[2]) # "PRK MMC Teneo HD")
            elif (cyl<-0.75) or (hoa>0.6):
                decision   = self._surgeries[2]       # "PRK MMC Teneo HD"
                acceptable.append(self._surgeries[1]) # "PRK MMC WL Allegretto")

        return decision, acceptable

    def _MyopicMedium(self,se,meanK,pachy,hoa,kpi,cyl,meanIridoAngle,age,acd,axialLength,reqCorD,priorLasik,pvd,opticalZoneDiameter,flapThickness,epithelial_pattern):
        """
          Surgery recommendation for medium myopic (-6<=Spherical equivalent<=-1.5 )

          Returns:
          ----------
            decision: str
              a decision based on the keys in self._surgeries
            acceptable: list (str)
              a list of acceptable surgeries from the list in
              self._surgeries, which are also an acceptable procedure
        """
        decision   = None
        acceptable = []
        # compute residual stromal bed
        rsb = self._ComputeRSB(pachy,reqCorD,opticalZoneDiameter,flapThickness)
        if (meanK<=46.5) and (pachy>=500) and (kpi<=10) and (rsb>300) and (epithelial_pattern.lower()=='normal'):
            decision = self._surgeries[7]# "Ultra Lasik. Femto + WFG Teneo HD"

            if (cyl>-0.75) and (hoa<0.6):
                if (se<=-1.5) and (se>-6):
                    if (age>=45):
                        acceptable.append(self._surgeries[5])#"Smile Femto Visumax")
                    else:
                        acceptable.append(self._surgeries[6])#"Femto Lasik. Femto +WL Allegretto")
            if rsb>400:
                if cyl>=-0.75 and hoa<0.6:
                    acceptable.append(self._surgeries[1])#"PRK MMC WL Allegretto")
                elif cyl<-0.75 or hoa>0.6:
                    acceptable.append(self._surgeries[2])# "PRK MMC Teneo HD")

            # elif (se<=-6) and (se>-8):
            #     if (age<45):
            #         decision = "Femto Lasik. Femto +WL Allegretto"
            # elif (cyl<-0.75) or (hoa>0.6):
            #     decision = "Ultra Lasik. Femto + WFG Teneo HD"
        elif (se>=-6):
            decision,acceptable = self._MyopicLow(se,cyl,hoa,pachy,reqCorD,opticalZoneDiameter,flapThickness)
        elif (se<-6):
            decision,acceptable = self._MyopicHigh(se,cyl,hoa,axialLength,acd,meanIridoAngle,age,priorLasik,pvd)
        return decision, acceptable

    def _MyopicHigh(self,se,cyl,hoa,axialLength,acd,meanIridoAngle,age,priorLasik,pvd):
        """
             Surgery recommender for high myopia: -24<spherical equivalent <-6

             Parameters:
             ----------
             se, float
              spherical equivalent (diopter)
             cyl, negative float
              cylinder (diopter)
             hoa, float
              high order aberrations
             axialLength, float
              axial length (mm)
             acd, float
              anterior chamber depth (mm)
             meanIridoAngle, float
              average irido angle (left right, for same eye)
             age, float
              years
             priorLasik, bool, default = False
              prior surgery (True/False)
             pvd, bool, default =False
              posterior vitreous detachmment

              Returns:
              --------

        """

        decision   = None
        acceptable = []
        if (acd>2.9) and (meanIridoAngle>28) and (axialLength<28.5) and (pvd==False):
            if (se<=-17) and (se>=-24):
                if (cyl>=-0.75) and (hoa<=0.6):
                    decision = self._surgeries[8]# "Bioptic (ICL +FemtoLasik)"
            elif (se<=-6) and (se>-17) and (age>=21) and (age<=55):
                if (cyl>-0.75):
                    decision = self._surgeries[9]  # "ICL Visian Spheric"
                elif (cyl<-0.75) or (hoa>0.6):
                    decision = self._surgeries[10] #"ICL Visian Toric"
        else:
            if (age>55) and pvd:
                if priorLasik:
                    decision = self._surgeries[11]# "Phaco +IOL/ Prelex" if priorLasik else "Phaco +Customized IOL"
        return decision, acceptable

    def _HyperopicRecommender(self,sphericalEq,cyl,meanK,pachy,kpi,age,acd,meanIridoAngle,priorLasik,pvd):
        """
         Surgery decision for cases of positive spherical equivalent
        """
        params = locals().copy()
        params.pop('self') # exclude self

        decision   = None
        acceptable = []
        flag = self._VerifyFunctionInputVariables(params,invalidVals=[None])
        if flag:
            if sphericalEq<=1.5:
                decision,acceptable = self._HyperopicLow(cyl)
            elif sphericalEq<=4.5:
                decision,acceptable = self._HyperopicMedium(sphericalEq,meanK,cyl,pachy,kpi,age)
            elif (sphericalEq>=4) and (sphericalEq<=12):
                decision,acceptable = self._HyperopicHigh(sphericalEq,cyl,age,acd,meanIridoAngle,priorLasik=priorLasik,pvd=pvd)
        return decision, acceptable

    def _HyperopicLow(self,cyl)->str:
        """
         Recommender for hyporopic case with low SE

         Parameters:
         -----------
         cyl : float
          cylinder (diopter)

         Returns:
         ----------
            decision: str
              a decision based on the keys in self._surgeries
            acceptable: list (str)
              a list of acceptable surgeries from the list in
                self._surgeries, which are also an acceptable procedure
        """
        decision   = None
        acceptable = []
        if cyl>-0.75:
            decision = self._surgeries[1]# "PRK MMC WL Allegretto"
        else:
            decision = self._surgeries[2]# "PRK MMC Proscan"
        return decision, acceptable

    def _HyperopicMedium(self,se:float,meanK:float,cyl:float,pachy:float,kpi:float,age:float)->str:
        decision   = None
        acceptable = []
        if (meanK<=45) and (pachy>=500) and (kpi<23):
            if (cyl>-0.75):
                if age>42:
                    decision = self._surgeries[14] # "PresbyLasik: Femto+WL -1.25Q=-0.8"
                else:
                    decision = self._surgeries[6]  # "FemtoLasik: Femto+WL Allegretto
            else:
                if age>42:
                    decision = self._surgeries[15] # "PresbyLasik: Femto +Teneo Supracor"
                else:
                    decision = self._surgeries[13] #  "FemtoLasik: Femteo + Teneo Proscan"
        elif se<=3.5:
            decision,acceptable = self._HyperopicLow(cyl)
        return decision, acceptable

    def _HyperopicHigh(self,se,cyl,age,acd,meanIridoAngle,priorLasik=False,pvd=False):
        """
         Decision for hyperopic case with high SE
        """

        decision   = None
        acceptable = []
        if (acd>3) and (meanIridoAngle>32):
            if (se>8) and (se<12):
                decision = self._surgeries[8]  # "Bioptic (ICL+FemtoLasik)"
            elif (se>4) and (se<8):
                if cyl>-0.75:
                    decision = self._surgeries[9]  # "ICL Visian Spheric"
                else:
                    decision = self._surgeries[10] #"ICL Visian Toric"
        else:
            if (age>55):
                if pvd:
                    decision = self._decision[11] # "Phaco+IOL/Prelex: Multifocal/Monovision"
                if priorLasik:
                    decision = self._decision[12] # "Phaco+Customized IOL"
        return decision, acceptable

    @staticmethod
    def _VerifyClassInputVariables(args):
        """
         Check if input variables are valid

         Parameters:
         -----------
         args: dictionary
           dictionary of input varible names and values

         Returns:
         -------
         flag: bool

         Raises:
         --------
         ValueError exception
         TypeError exception
        """
        def _raiseIfNegative(key,val,strict=False):
            if strict:
                if val<0:
                    raise ValueError(f"{key} must be positive. Got {val}")
            else:
                if val<=0:
                    raise ValueError(f"{key} must be strictly positive. Got {val}")

        def _raiseIfPositive(key,val,strict=False):
            if strict:
                if val>0:
                    raise ValueError(f"{key} must be negative. Got {val}")
            else:
                if val>=0:
                    raise ValueError(f"{key} must be strictly negative. Got {val}")

        def _raiseIfNotType(key,val,valType):
            if not isinstance(val,valType):
                raise TypeError(f"{key} must be of type {valType}. Got {type(val)}")

        for kIdx in args.keys():
            if kIdx=='cyl':
                _raiseIfPositive(kIdx,args[kIdx],strict=True)
            elif kIdx in ['acd','axialLength','meanK','age','pachy','meanIridoAngle']:
                _raiseIfNegative(kIdx,args[kIdx])
                # check range
                if kIdx=='meanIridoAngle':
                    if args[kIdx]>60:
                        raise ValueError('irido angle too large. Please check value')
            elif kIdx in ['kpi','hoa']:
                _raiseIfNegative(kIdx,args[kIdx],strict=True)
                if kIdx=='kpi':
                    if args[kIdx]>100:
                        raise ValueError(f"kpi must be positive between 0 and 100. Got {args[kIdx]}")
            elif kIdx=='dominantEye':
                _raiseIfNotType(kIdx,args[kIdx],str)
                if args[kIdx] not in ['left','right']:
                    raise ValueError(f"dominant eye must be either left or right. Got {args[kIdx]}")
            elif kIdx in ['priorLasik','pvd']:
                _raiseIfNotType(kIdx,args[kIdx],bool)

    @staticmethod
    def _VerifyFunctionInputVariables(localVar,invalidVals=[None]):
        """
            Check if input variable to a function are valid

            Parameters:
            ----------
            localVar, dictionary
                input local variables to a function, key=variable name

            Output
            -------
            flag: bool
             True if all vriables are valid (not in invalidVals)
             False if at least one variable is in invalidVals
        """
        flag = True
        if isinstance(localVar,dict):
            for kIdx in localVar:
                if localVar[kIdx] in invalidVals:
                    flag = False
        else:
            raise ValueError('localVar must be a dictionary')
        return flag




