import numpy as np
import pandas as pd

class Predictor:
    """
        Predict abnormality risks based on Objective vx120/130 data
    """

    def Predict(self,data):
        """
            Predict risks based on objective data

            Parameters:
            ----------
            data : DataFrame
                input dataframe, the objective parsed measurements of vx120 machine after parsing and transforming
                thus, data=vxTransformer.Transform(vxParser.Parse(vx120zipfile)
                The risk computed are:
                * Keratoconus risk
                * Pachymetry risk
                * Tonometry risk
                * Angle closure risk

            Output:
            --------
            results : DataFrame
                the input dataFrame with appended fields names starting with Predicted_Risk
                e.g. Predicted_Risk_Angle_Closure_Right,
                     Predicted_Risk_Tono_Left

        """
        assert data.__class__==pd.DataFrame, f"Input data must be a pandas.DataFrame, recieved {data.__class__}"
        results = pd.DataFrame(index=data.index)
        for lIdx in ['_Left','_Right']:
            results[f'Predicted_Risk_Keratoconus{lIdx}'] = data[f'Topo_KERATOCONUS_Kpi{lIdx}'].apply(self.ComputeKeratoconusRisk)
            results[f'Predicted_Risk_Pachy{lIdx}']       = data[f'Pachy_MEASURE_Thickness{lIdx}'].apply(self.ComputePachyRisk)

        for ind in data.index:
            for lIdx in ['_Left','_Right']:
                age    = data.loc[ind,'Age']
                # kpi    = data.loc[ind,f'Topo_KERATOCONUS_Kpi{lIdx}']
                pachy  = data.loc[ind,f'Pachy_MEASURE_Thickness{lIdx}']
                iopc   = data.loc[ind,f'IOPcorrected{lIdx}']
                acv    = data.loc[ind,f'AnteriorChamberVol{lIdx}']
                acd    = data.loc[ind,f'Pachy_MEASURE_Acd{lIdx}']
                irAngR = data.loc[ind,f'Pachy_MEASURE_IridoAngleR{lIdx}']
                irAngL = data.loc[ind,f'Pachy_MEASURE_IridoAngleL{lIdx}']
                se     = (data.loc[ind,f'WF_SPHERE_R_3{lIdx}']+data.loc[ind,f'WF_CYLINDER_R_3{lIdx}']/2)

                # results.loc[ind,f'Predicted_Risk_Keratoconus{lIdx}'] = self.ComputeKeratoconusRisk(kpi)
                # results.loc[ind,f'Predicted_Risk_Pachy{lIdx}']       = self.ComputePachyRisk(pachy)
                results.loc[ind,f'Predicted_Risk_Tono{lIdx}']        = self.ComputeTonoRisk(iopc,pachy,age)
                results.loc[ind,f'Predicted_Risk_AngleClosure{lIdx}']= self.ComputeAngleClosureRisk(age,acv,acd,irAngL,irAngR,se)
        return results

    @staticmethod
    def ComputeKeratoconusRisk(kpi):
        if pd.notna(kpi):
            if kpi<0.3:
                return "Low"
            else:
                return "High"
            # return np.where(kpi<0.3,"Low","High")
        else:
            return pd.NA

    @staticmethod
    def ComputePachyRisk(pachy):
        if pd.notna(pachy):
            if (pachy<490) or (pachy>600):
                return 'Abnormal'
            elif (pachy>=520) and (pachy<=580):
                return 'Normal'
            elif (pachy>=490 and pachy<520) or (pachy>580 and pachy<=600):
                return 'Mild'
        else:
            return pd.NA

    @staticmethod
    def ComputeTonoRisk(iopc,pachy,age):
        """
         Parametes:
         ----------
         iopc, float
          tonometry corrected values (see IOPc method)
         pachy, float
          pachimetry (mu m)
         age, float
           patient age (years) at time of examination
         Output:
         -------
          risk, str
             low, medium, or high
        """
        if pd.notna(iopc) and pd.notna(pachy) and pd.notna(age):
            score = 0
            if age<=50:
                # iop corrected
                if iopc<=17:
                    score+=1
                elif (iopc>17) and (iopc<=22):
                    score+=2
                elif (iopc>20):
                    score+=3
                # pachy
                if pachy>560:
                    score+=1
                elif pachy>500 and pachy<=560:
                    score+=2
                elif pachy<=500:
                    score+=3

            elif age>50:
                # iop corrected
                if iopc<=20:
                    score+=1
                elif iopc>20 and iopc<=24:
                    score+=2
                elif iopc>24:
                    score+=3
                # pachy
                if pachy>=560:
                    score+=1
                elif pachy>=500 and pachy<560:
                    score+=2
                elif pachy<500:
                    score+=3
            if score<=2:
                return "Low"
            elif score>2 and score<=4:
                return "Medium"
            elif score>4:
                return "High"
        else:
            return pd.NA

    @staticmethod
    def ComputeAngleClosureRisk(age,anteriorChamberVol,acd,iridoAngleL,iridoAngleR,sphericalEquivalent):
        """
         Computation of the risk of angle clusure (Glaucoma risk)

         Parameters:
         -----------
         age, float
          age of patient at time of measurements (years)
         anteriorhamberVol, float, non-negative
          anterior chambeer volume (see vx120TRansformer for computation)
         acd,  float non negative
           anterior chamber depth
         iridoAngleL, float
          left irido angle (degrees)
         iridoAngleR, float
          right irido angle (degrees)
         sphericalEquivalen, float
          spherical equivalent (D)

         Returns:
         ---------
          risk, str
           either Low, Medium, or High
        """

        if pd.notna(age) and age>0 and\
           pd.notna(anteriorChamberVol) and \
           anteriorChamberVol>0 and \
           pd.notna(acd) and \
           acd>0 and \
           pd.notna(iridoAngleL) and \
           iridoAngleL>0 and \
           pd.notna(iridoAngleR) and \
           iridoAngleR >0 and \
           pd.notna(sphericalEquivalent):

            score = 0
            if age<50:
                score+=1
            elif age>=50 and age<60:
                score+=2
            elif age>=0:
                score+=3

            if anteriorChamberVol>=110:
                score+=1
            elif anteriorChamberVol<=110 and anteriorChamberVol>100:
                score+=2
            elif anteriorChamberVol<100:
                score+=3

            if acd >=2.2:
                score+=1
            elif acd>=1.8 and acd<2.2:
                score+=2
            elif acd<1.8:
                score+=3

            if sphericalEquivalent>2 and sphericalEquivalent<=3:
                score+=1
            elif sphericalEquivalent<=4 and sphericalEquivalent>3:
                score+=2
            elif sphericalEquivalent>4:
                score+=3

            if score<=5:
                return "Low"
            elif score>5 and score<=10:
                return "Medium"
            elif score>10:
                return "High"
        else:
            return pd.NA