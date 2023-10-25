import os
import re
import dateutil
import datetime
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RANSACRegressor
import joblib

from autorefeyelib.IOLpower.Formulas import Formulas

class Predictor:
    """
     Classify IOL formulas according to their accuracy given a set of iolMaster measurements
    """

    def __init__(self):
        """
        """
        # self.model       = joblib.load(os.path.join(os.path.dirname(__file__),'model','iolClassifier.sav'))
        self.data        = pd.read_csv(os.path.join(os.path.dirname(__file__),'data','EMRIolMasterJointDB_2021_08_04.csv'))
        self._regChooser = joblib.load(os.path.join(os.path.dirname(__file__),'model','reg_chooser.sav')) # load regressor classifier
        self._regressors = joblib.load(os.path.join(os.path.dirname(__file__),'model','regressors.sav'))  # load regressors
        self.formulas    = Formulas()
        # self._logger   = self._GetLogger()
        # compute all needed constant for computation of the IOL formulas
        self._ComputeConstantsAndMeans()
        # self._logger.info("IOL predictor class successfully initialized")

    def _ComputeConstantsAndMeans(self,refractionIndex=1.335,dioptricDelta=0.25,vertexDistance=0.012):
        #TODO: move to DB/Formulas
        # combine left and right eye data from the joint EMR iolMaster DB
        Rc          = (self.data['l_radius_se_mean'].append(self.data['r_radius_se_mean']))
        meanK       = (refractionIndex-1)*1000/Rc
        axialLength = self.data['l_axial_length_mean'].append(self.data['r_axial_length_mean'])
        acd         = self.data['l_acd_mean'].append(self.data['r_acd_mean'])
        wtw         = self.data['l_wtw_mean'].append(self.data['r_wtw_mean'])
        Rt          = self.data['IolDesiredRefractionPostOp_Left'].append(self.data['IolDesiredRefractionPostOp_Right'])
        Rf          = self._Round(self.data['VisualAcuitySphere_Left'].append(self.data['VisualAcuitySphere_Right']),dioptricDelta)
        t           = self._Round(self.data['IolTheoreticalPower_Left'].append(self.data['IolTheoreticalPower_Right']),dioptricDelta)
        # Pi          = self._Round(self.data['IolDesiredRefractionPostOp_Left'].append(self.data['IolDesiredRefractionPostOp_Right']),dioptricDelta)
        Pi          = self._Round(self.data['IolPowerImplanted_Left'].append(self.data['IolPowerImplanted_Right']),dioptricDelta)
        age         = self.data['Age'].append(self.data['Age'])
        # remove invalid entries
        inds = np.where( (meanK.isna())\
                        | (meanK<=0)\
                        | (axialLength<=0)\
                        | (acd<=0)\
                        | (age<=0) \
                        | (axialLength.isna())\
                        | (acd.isna())\
                        | (Rt.isna()))[0]
        index = Rc.index[inds]
        Rc.drop(index,inplace=True)           # mean cornea radius (mm)
        meanK.drop(index,inplace=True)        # mean keratometry (D)
        axialLength.drop(index,inplace=True)  # axial length (mm)
        acd.drop(index,inplace=True)          # anterior chamber depth (mm)
        wtw.drop(index,inplace=True)          # white to white (mm)
        Rt.drop(index,inplace=True)           # target refraction (D)
        Rf.drop(index,inplace=True)           # final refraction (D)
        t.drop(index,inplace=True)            # theoretical power (D)
        Pi.drop(index,inplace=True)           # power implanted (D)
        age.drop(index,inplace=True)          # age (years)
        # compute means
        self._averages = {'ACD':acd.mean(),
                          'WTW':wtw.mean(),
                          'AxialLength':axialLength.mean(),
                          'Age':age.mean(),
                          'MeanK':meanK.mean(),
                          'CornealHeight':self.formulas.MeanCornealHeight(wtw,Rc,axialLength)}
        self._constants = {'HolladaySurgeonFactor':self.formulas.HolladaySurgeonFactor(meanK,axialLength,Pi,Rf,v=vertexDistance*1000),
                           'HofferQPersonalizedACD':self.formulas.HofferQPersonalisedACD(meanK,axialLength,Pi,Rf,n_c=1.336,v=vertexDistance),
                           'HaigisELP':self.formulas.HaigisELP(meanK,acd,axialLength,Pi,Rf,na=1.3315,v=vertexDistance)[0]}
        # self._meanACD                = acd.mean()
        # self._meanWTW                = wtw.mean()
        # self._meanAxialLength        = axialLength.mean()
        # self._meanAge                = age.mean()
        # self._meanK                  = meanK.mean()
        # self._meanCornealHeight      = self.formulas.MeanCornealHeight(wtw,Rc,axialLength)
        # compute constants from retrospective data
        # self._surgeonFactor          = self.formulas.HolladaySurgeonFactor(meanK,axialLength,Pi,Rf,v=vertexDistance*1000)
        # self._hofferQPersonalizedACD = self.formulas.HofferQPersonalisedACD(meanK,axialLength,Pi,Rf,n_c=1.336,v=vertexDistance)
        # self._haigisELP,_,_,_        = self.formulas.HaigisELP(meanK,acd,axialLength,Pi,Rf,na=1.3315,v=vertexDistance)

    def PredictIolPower(self,measurements,Aconst=118.9,targetRefraction=0,pDelta=0.5, rDelta=0.25):
        """
         This function serves as an entry point for prediction using the parsed
         vx120 and Revo data
         Prediction can also be called directly with input variables using _Predict()

         Parameters:
         -----------
         measurements, DataFrame
          dataframe containing the vx120 parsed data and revo parsed data
         Aconst, float, default=118.9
          A- constant of the lens
         targetRefraction, float, default = 0,
          target refraction (sphere only) after surgery
         pDelta, float, default =0.5,
           dioptric delta of the IOL predicted, the value predicted will be
           rounded to the nearest pDelta
         rDelta, float, default=0.25
           dioptric delta for predicted final refraction
           the predicted values wllll be rounded to the nearest rDelta

         Output:
         ----------
         results, DataFrame
          input measurements with added Predicted IOL fields
          Predicted_IOlpowere_Left(Right), float (D)
          Predicted_IolFinalRefraction_Left(Right), float (D)
          Predicted_IolFormula_Left(right), str
        """
        Aconst       = np.asanyarray(Aconst)
        ind          = measurements.index
        predictedIOL = pd.DataFrame(index=ind)

        predictedIOL.loc[ind,'Aconst'] = Aconst
        for lIdx in ['_Left','_Right']:
            # Prepare input
            dataIn                = pd.DataFrame(index=ind)
            dataIn['Aconst']      = Aconst
            if 'Age' in measurements.columns:
                dataIn['age'] = measurements['Age']
            else:
                dataIn['age'] = self._averages['Age']
                # self._logger.warn(f'Age is missing from input feature vactor. assigning mean value {self._meanAge}')
            if (f'Topo_Sim_K_K1{lIdx}' in measurements.columns) and (f'Topo_Sim_K_K2{lIdx}' in measurements.columns):
                dataIn['meanK']       = 0.5*(measurements[f'Topo_Sim_K_K1{lIdx}'] + measurements[f'Topo_Sim_K_K2{lIdx}'])
            else:
                dataIn['meanK'] = self._averages['MeanK']
            if f'Pachy_MEASURE_Acd{lIdx}' in measurements.columns:
                dataIn['ACD']   = measurements[f'ACD_Avg{lIdx}']-measurements[f'CCT_Avg{lIdx}']
                # dataIn['ACD']   = measurements[f'Pachy_MEASURE_Acd{lIdx}']
            else:
                dataIn['ACD'] = self._averages['ACD']
            if f'Pachy_MEASURE_WhiteToWhite{lIdx}' in measurements.columns:
                dataIn['WTW'] = measurements[f'Pachy_MEASURE_WhiteToWhite{lIdx}']
            else:
                dataIn['WTW'] = self._averages['WTW']
            if f'AxialLength_Avg{lIdx}' in measurements.columns:
                dataIn['axialLength'] = measurements[f'AxialLength_Avg{lIdx}']
            else:
                dataIn['axialLength'] = self.averages['AxialLength']

            dataIn['targetRefraction'] = targetRefraction

            pIOL,rIOL,formula = self._Predict(dataIn['Aconst'],
                                                dataIn['age'],
                                                dataIn['meanK'],
                                                dataIn['ACD'],
                                                dataIn['WTW'],
                                                dataIn['axialLength'],
                                                dataIn['targetRefraction'])

            predictedIOL.loc[ind,f'Predicted_IOL_Power{lIdx}']           = self._Round(pIOL,pDelta)
            predictedIOL.loc[ind,f'Predicted_IOL_FinalRefraction{lIdx}'] = self._Round(rIOL,rDelta)
            predictedIOL.loc[ind,f'Predicted_IOL_Formula{lIdx}']         = formula


        # self._logger.info(f' IOL power predicted. A-const= {Aconst} Rt={targetRefraction}')
        return predictedIOL

    def _Predict(self,Aconst,age,meanK,acd,wtw,axialLength,targetRefraction):
        """
            Predict the IOL power needed to be implanted to reach target refraction
            Parameters:
            ------------
            Aconst- float
             Manufactorer A constant for the IOL
            age, float
             age of the patient (years)
            meanK, float
             average keratometry (D)
            acd, float
             anterior chamber depth (mm)
            wtw, float
             white to white (mm)
            axialLength, float
             axial length (mm)
            target refraction, float
             the expected refraction post surgery (D)
            Output:
            -------
            P, float
             IOl power predicted to reach target refraction (D)
            R, float
             refraction predicted after surgery (D)
            formula, str
                the type of formula used to obtain P and R
        """

        # Complete missing values with means
        # TODO: move imputing to data preparation ( dedicated Input loader)
        age              = np.where(pd.isna(age),self._averages['Age'],age)
        axialLength      = np.where(pd.isna(axialLength),self._averages['AxialLength'],axialLength)
        Aconst           = np.where(pd.isna(Aconst),118.9,Aconst) # set default A const whetre missing
        meanK            = np.where(pd.isna(meanK),self._averages['MeanK'],meanK)
        acd              = np.where(pd.isna(acd),self._averages['ACD'],acd)
        wtw              = np.where(pd.isna(wtw),self._averages['WTW'],wtw)
        targetRefraction = np.where(pd.isna(targetRefraction),0,targetRefraction)

        # get predicted IOL power and refraction from all IOL formulas
        features                           = pd.DataFrame()
        features.loc[:,'age']              = age
        features.loc[:,'meanK']            = meanK
        features.loc[:,'ACD']              = acd
        features.loc[:,'WTW']              = wtw
        features.loc[:,'axialLength']      = axialLength
        features.loc[:,'targetRefraction'] = targetRefraction
        # meanCornealHeight,meanACD,surgeonFactor,hofferQPersonalizedACD
        P,R,E,_ = self.formulas.ComputeAllFormulas(["SRKT","Shammas","Haigis","Holladay-1","Binkhorst-2","Hoffer-Q","Olsen"],
                                                   Aconst,
                                                   features['meanK'],
                                                   features['ACD'],
                                                   features['WTW'],
                                                   features['axialLength'],
                                                   features['targetRefraction'],
                                                   self._averages['CornealHeight'],
                                                   self._averages['ACD'],
                                                   self._constants['HolladaySurgeonFactor'],
                                                   self._constants['HofferQPersonalizedACD'])

        formulaClass = self.model.predict(features)
        formulaName  = P.columns[formulaClass]

        # reg_pred      = self._regChooser.predict(features)


        return P[formulaName].values, R[formulaName].values,formulaName.values

    @staticmethod
    def _Round(val,delta):
        """
         Round value to the nearese delta
        """
        if not isinstance(delta,(float,int)):
            raise ValueError('delta must be a float, got {delta.__class__}')
        if delta<0 :
            raise ValueError('Delta must be between 0 and 1, got {delta}')
        return np.round(val/delta)*delta

    def PredictdP(self,Aconst, measurements, targetRefraction,pDelta =0.5, rDelta=0.25):
        """
         Predict dR (refraction delta) based on patient-specific parameters
         Use the predicted dR to compute the power delta dP and compute the new and corrected IOL power P.
         This function first predicts the formula to be used,
         it then computes the dR based on the selected trained regressor
         and computed the IOL power P using the selected formula.
         And finally uses the dR from the selected regressor to compute dP.
         The new predicted power is then P+dP.

         Parameters:
         ---------
         Aconst, float
           lens A constant
         measurements, pd.DataFrame
           patient measurements, transformed vx120 data (see vx120Transformer)
         targetRefraction, float
          targert refraction post surgery (D)
         pDelta, float, default =0.5
            round predicted power to nearest pDelta interval
         rDelta, float, default =0.25
            round predicted refraction to nearest rDelta interval


         Returns:
         ---------
         dp, array float
          predicted deltaP such that P_new = P+deltaP, where P is the predicted P
         dr, array float,
          predicted deltaR, where deltaR is R_pred-R_f, with R_pred the final refraction
          from a given formula after rounding the IOL power to nearest diopteric delta
        """
        if isinstance(Aconst, (pd.DataFrame, pd.Series, float, np.ndarray,list,int)):
            Aconst       = np.asanyarray(Aconst)
        else:
            raise ValueError(f'Aconst must be an array of floats instead got {Aconst.__class__}')
        if isinstance(measurements,pd.DataFrame):
            ind          = measurements.index
        else:
            raise ValueError(f'measurement must be of type  pandas.DataFrame, instead got {measurements.__class__}')
        # if len(Aconst)>1:
        #     if len(Aconst)!=len(measurements):
        #         raise ValueError('Aconst must either be a single value or an array the same length as measurements')
        predictedIOL = pd.DataFrame(index=ind)
        features     = ["age","axialLength","ACD","meanK","targetRefraction"]
        predictedIOL.loc[ind,'Aconst'] = Aconst
        missingList  = []
        # check input dataframe validdity

        # check input parameters' validaity
        for lIdx in ['_Left','_Right']:
            # Prepare input
            dataIn                     = pd.DataFrame(index=ind)
            dataIn['Aconst']           = Aconst
            dataIn['targetRefraction'] = targetRefraction

            if 'Age' in measurements.keys():
                dataIn['age'] = measurements['Age']
                dataIn['age'].replace([pd.NA,np.nan,None],self._averages['Age'],inplace=True)
            else:
                dataIn['age'] = self._averages['Age']
                missingList.append('Age')
            if (f'Topo_Sim_K_K1{lIdx}' in measurements.keys()) and (f'Topo_Sim_K_K2{lIdx}' in measurements.keys()):
                dataIn['meanK']       = 0.5*(measurements[f'Topo_Sim_K_K1{lIdx}'] + measurements[f'Topo_Sim_K_K2{lIdx}'])
                dataIn['meanK'].replace([pd.NA,np.nan,None],self._averages['MeanK'],inplace=True)
            else:
                dataIn['meanK'] = self._averages['MeanK']
                missingList.append('Topo_Sim_K_K1{lIdx}')
            if f'Pachy_MEASURE_Acd{lIdx}' in measurements.keys() and f'CCT_Avg{lIdx}' is measurements.keys():
                dataIn['ACD']   = measurements[f'ACD_Avg{lIdx}']-measurements[f'CCT_Avg{lIdx}']
                dataIn['ACD'].replace([pd.NA,np.none,None],self._averages['ACD'],inplace=True)
            else:
                dataIn['ACD'] = self._averages['ACD']
                missingList.append('Pachy_MEASURE_Acd{lIdx}')

            if f'Pachy_MEASURE_WhiteToWhite{lIdx}' in measurements.keys():
                dataIn['WTW'] = measurements[f'Pachy_MEASURE_WhiteToWhite{lIdx}']
                dataIn['WTW'].replace([pd.NA,np.nan,None],self._averages['WTW'],inplace=True)
            else:
                dataIn['WTW'] = self._averages['WTW']
                missingList.append(f'Pachy_MEASURE_WhiteToWhite{lIdx}')

            if f'AxialLength_Avg{lIdx}' in measurements.keys():
                dataIn['axialLength'] = measurements[f'AxialLength_Avg{lIdx}']
                dataIn['axialLength'].replace([pd.NA,np.nan,None],self._averages['AxialLength'],inplace=True)
            else:
                dataIn['axialLength'] = self._averages['AxialLength']
                missingList.append(f'AxialLength_Avg{lIdx}')



            formulas = list(self._regressors.keys()) # get formuals by names of regressors loaded
            P,R,E,L = self.formulas.ComputeAllFormulas(formulas,
                                                    Aconst,
                                                    dataIn['meanK'],
                                                    dataIn['ACD'],
                                                    dataIn['WTW'],
                                                    dataIn['axialLength'],
                                                    dataIn['targetRefraction'],
                                                    self._averages['CornealHeight'],
                                                    self._averages['ACD'],
                                                    self._constants['HolladaySurgeonFactor'],
                                                    self._constants['HofferQPersonalizedACD'],
                                                    pDelta = pDelta)

            # featMat_class = dataIn[self._regChooser.feature_names_in_]
            # choose which regressor (formula) to use
            reg_pred = self._regChooser.predict(dataIn.loc[ind,self._regChooser.feature_names_in_])[0]
            # get the name of the predicted formula
            formula  = formulas[reg_pred]
            # predict the dr according to the chosen regressor
            # update the ACd and axial length according to the prediction of the chosen formula
            dataIn.loc[ind,'ACD']         = E.iloc[0][formula]
            dataIn.loc[ind,'axialLength'] = L.iloc[0][formula]
            dr  = self._regressors[formula].predict(dataIn.loc[ind,self._regressors[formula].feature_names_in_])
            # compute the ELP, axial length of the chosen formula
            # print(f'Predicted class {reg_pred[rIdx]}, regressor {reg_keys[reg_pred[rIdx]]}')
            n_c = self.formulas.GetParams()[formula]["n_c"]
            dp = self.deltaP(n_c,
                             dataIn['ACD'],
                             dataIn['meanK'],
                             targetRefraction,
                             dr)
            # compute the new (corrected) IOL power
            predictedIOL.loc[ind,f'Predicted_IOL_Power{lIdx}']                = self._Round(P[formula].values + dp.values,pDelta)
            predictedIOL.loc[ind,f'Predicted_IOL_FinalRefraction{lIdx}']      = self._Round(R[formula].values,rDelta)
            predictedIOL.loc[ind,f'Predicted_IOL_subjectiveRefraction{lIdx}'] = self._Round(targetRefraction+dr,rDelta)
            predictedIOL.loc[ind,f'Predicted_IOL_Formula{lIdx}']              = formula
            # print(f' Target refraction {lIdx}: {targetRefraction:.2f}, predicted dr {dr[0]:.2f}, predicted subjective using P={P.loc[ind,formula].values[0]:.2f} is {targetRefraction+dr[0]:.2f}, new P= {self._Round(P.loc[ind,formula] + dp,pDelta).values[0]}')

        return predictedIOL

    @staticmethod
    def deltaP(n_c,elp,meanK,Rt,dR):
        """
         Compute the change in power based on the change of refraction

         Parameters:
         ----------
         n_c, float,
          refraction index of cornea
         elp, float
          effective lens position (mm)
         meanK, float
          average keratometry (D)
         Rt, float
          target refraction (D)
         dR, float
          refraction delta

         Returns
         --------
         dP, float
           power delta (D)
        """
        alpha = n_c-(elp/1000)*(meanK+Rt)
        return -(n_c**2)*dR/(alpha*(alpha-(elp/1000)*dR))

    def ComputeFormula(self,fName,params):
        if fName.lower()=='srkt':
            P,R,elp,al = self.formulas.SRKT(params['Aconst'],params['meanK'],params['axialLength'],params['Rt'])
        elif fName.lower()=='t2':
            P,R,elp,al = self.formulas.SRKT(params['Aconst'],params['meanK'],params['axialLength'],params['Rt'],T2=True)

    @staticmethod
    def RegressionLoss(gt_vals,pred_vals,rDelta=0.25, residual_threshold=1.5):
        """
         Loss function for RANSACregression methodologies. Used by regressors saved and loaded

         Parameters:
         -----------
         gt_vals, array float,
            1d array of ground truth values
         pred_vals, array float
            1d array same length as gt_vals, with predicted values
         rDelta, float, default=0.25
            dioptric interval for refraction
         residual_threshold, float, default = 1.5
            threshold above which the loss will be considered an outlier in the RANSAC fitting procedure

         Returns:
         ---------
         loss, array float
            loss for each observation based on error intervals of 0, 0.25, 0.5, 0.75,...
        """
        # compute loss
        loss     = np.ones(len(gt_vals))*residual_threshold
        abs_diff = abs(gt_vals-(pred_vals/rDelta).round()*rDelta)

        # loss[abs_diff==0]    = 0
        loss[abs_diff<=0.25] = 0
        loss[abs_diff==0.5]  = 0.5*residual_threshold
        loss[abs_diff>0.5]   = 2*residual_threshold
        # print(loss)
        return loss
