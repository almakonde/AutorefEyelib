import os
import re
from dateutil import parser as dateParser
from datetime import datetime
import pandas as pd
import numpy as np

"""
    Transform parsed vx120/130 data. The output of this class is used as input to Predictors and Reporting
"""
class Transformer:

    def __init__(self):
        # Load all possible keys
        self._vxKeys= list((pd.read_csv(os.path.join(os.path.dirname(__file__),'data','vx130Keys.csv'))).T.values[0])

    def Transform(self,data,vertexDistance=0.012,refractiveIndex=1.3375,diopterInterval=0.25,iopc_method="doughty_normal",min_age_years=3):
        """
             Transform the vx120/130 raw data to representable values
             Compute additional values which appear in the vx130 xml but not in the raw data
             such as the IOP corrected (IOPc) and the anterior chamber volume.
             Ocular wavefront is transformed and HOA, LOA, Coma, Trefoil are computed

             Parameters:
             -----------
             data, pd.DataFrame or pd.Series
                Parsed vx120/130 data
                (see vx120Parser, the output of Parser.ParseFolder() and Parser.ParseVXFromZip())

             Output:
             --------
             data, pd.DataFrame
                 a transformed dataframe, with added fields and rounded values
        """
        # first make sure indices are unique
        flag = True
        if '_isTransformed' in data.columns:
            if data['_isTransformed'].any():
               print('[vxTransformer] Input data has already been transformed')
               flag = False
            elif (data['_isTransformed']==False).all():
                flag = True
        if flag:
            orig_inds  = data.index
            data.index = range(len(data))
            self._VerifyInput(data)
            self._PopulateMissingFields(data)
            self._WavefrontTopoAnalysis(data)
            self._WavefrontAnalysis(data)
            self._TransformTopo(data,diopterInterval=diopterInterval,refractiveIndex=refractiveIndex)
            self._TransformTono(data,refractiveIndex=refractiveIndex,iopc_method=iopc_method)
            self._TransformPachy(data)
            self._TransformWavefront(data,diopterInterval=diopterInterval,vertexDistance=vertexDistance)
            self._TransformPatientCredentials(data,min_age_years=min_age_years)

            data['_isTransformed'] = True
            print('[Info][vx120Transformer] Done')
            # return the original indices
            data.set_index(orig_inds)
        return data

    def _VerifyInput(self,data):
        """ verify input is in the correct format """
        if isinstance(data,pd.Series):
            return data.to_frame().T # make sure a row dataframe is used
        elif not isinstance(data, pd.DataFrame):
            ValueError(f'input data must be either pandas.DataFrame or a pandas.Series, got {data.__class__}')

    def _PopulateMissingFields(self,data):

        # Assign nan to missing fields
        for kIdx in self._vxKeys:
            if kIdx not in data.keys():
                data[kIdx] = np.nan
        # replace faulty measurements with nan
        data.replace(to_replace=-1000,value=np.nan,inplace=True)
        return data

    def _TransformPatientCredentials(self,data,min_age_years=3):
        """
            Transofrom patient credentials: age, birthdate, gender, examdate and set ID
            dates are transformed to format dd/mm/yyyy (with leading zeros where possible)

            Parameters:
            -------------
            data (DataFrame) :
              parsed vx120/130 database

            Returns:
            ----------
            data (DataFrame):
             input data with parsed patient credientials

        """
        if 'Gender' in data.keys():
            data.loc[data.index,'Gender'] = data['Gender'].apply(self.AssignGender)
        elif 'Sex' in data.keys():
            data.loc[data.index,'Gender'] = data['Sex'].apply(self.AssignGender)

        data.loc[data.index,'ExamDate']  = data['CurrentDate'].apply(self.ParseDateToStr,dayfirst=True) if 'CurrentDate' in data.keys() else ''
        data.loc[data.index,'BirthDate'] = data['BirthDate'].apply(self.ParseDateToStr,dayfirst=True)  if 'BirthDate' in data.keys() else ''

        if 'BirthDate' in data.keys() and 'ExamDate' in data.keys():
            k = data['ExamDate'].apply(pd.to_datetime,dayfirst=True)-data['BirthDate'].apply(pd.to_datetime,dayfirst=True)
            data.loc[data.index,'Age'] = (k
                            .apply(self._ParseDateDiff,output='years')
                            .apply(self.Round,delta=0.1,ndigits=1)
                            )
        else:
            data['Age'] = np.nan

        ids  = (data['Surname']+data['Firstname']+data['BirthDate']+data['ExamDate']).apply(self._ToID)
        data.loc[data.index,'ID'] = ids
        data.set_index(ids.values,inplace=True) # set the ID as the index
        return data

    def _TransformTopo(self,data,diopterInterval=0.25,refractiveIndex=1.3375):
        """
            Transform values related to Topography
            Parameters:
            ----------
            data, DataFrame
                parsed vx120/130 measurements
            dioptricInterval, float, default =0.25
              rounded values will be rounded to nearest dioptricInterval
            refractiveIndex, float, default = 1.3375
               refractive index for the vx machine,
               to compute cornea radius from keratometry, and vice versa

            Returns:
            --------
            data, DataFrame,
             transformed data
        """
        for e in ['_Left','_Right']:
            data.loc[:,f'Topo_Sim_K_Cyl{e}'] = (data[f'Topo_Sim_K_Cyl{e}']
                                        .apply(self.Round,args=[diopterInterval,2])
                                        )
            data.loc[:,f'Topo_Sim_K_Avg{e}'] = (data[f'Topo_Sim_K_Avg{e}']
                                        .apply(self.Round,args=[diopterInterval,2])
                                        )
            data.loc[:,f'Topo_KERATOCONUS_Kpi{e}'] = (data[f'Topo_KERATOCONUS_Kpi{e}']
                                                .apply(self.Round,args=[1,0])
                                                )
            data.loc[:,f'Topo_GENERAL_Geo_e{e}']   = (data[f'Topo_GENERAL_Geo_e{e}']
                                                .apply(self.Round,args=[0.01,2])
                                                )
            data.loc[:,f'Topo_GENERAL_Geo_P{e}'] = (data[f'Topo_GENERAL_Geo_P{e}']
                                             .apply(self.Round,args=[0.01,2])
                                             )
            # Compute the Q
            data.loc[:,f'Topo_GENERAL_Geo_Q{e}'] = (-data[f'Topo_GENERAL_Geo_e{e}']**2).apply(self.Round,args=[0.01,2])

            data.loc[:,f'Topo_TOPO_PD{e}'] = (data[f'Topo_TOPO_PD{e}']
                                        .apply(self.Round,args=[0.1,1])
                                        ) #if f'Topo_TOPO_PD{e}' in data.keys() else np.nan
            for mIdx in ['Photo','Meso','7']:
                data.loc[:,f'Topo_Diameter_{mIdx}{e}'] = (2*data[f'Topo_RadiusAperture_{mIdx}{e}']).apply(self.Round,args=[0.1,1])

            for kIdx in [1,2]: # translate from mm to Diopters
                data.loc[:,f'Topo_Sim_K_K{kIdx}{e}'] = (data[f'Topo_Sim_K_K{kIdx}{e}']
                                                    .apply(self.mm_D,args=[refractiveIndex])
                                                    .apply(self.Round,args=[diopterInterval,2])
                                                    )
                data.loc[:,f'Topo_Sim_K_K{kIdx}_axis{e}'] = data[f'Topo_Sim_K_K{kIdx}_axis{e}'].apply(self.Round,args=[1,0])

            # TODO: move to Others
            data.loc[:,f'kMax{e}']              = data[[f'Topo_Sim_K_K1{e}',f'Topo_Sim_K_K2{e}']].max(axis=1)
            data.loc[:,f'Radius_Cornea_R1{e}']  = (data[f'Topo_Sim_K_K1{e}']
                                            .apply(self.mm_D,args=[refractiveIndex])
                                            .apply(self.Round,args=[0.1,1]))
            data.loc[:,f'Radius_Cornea_R2{e}']  = (data[f'Topo_Sim_K_K2{e}']
                                            .apply(self.mm_D,args=[refractiveIndex])
                                            .apply(self.Round,args=[0.1,1]))
            data.loc[:,f'Radius_Cornea_Avg{e}'] = 0.5*(data[f'Radius_Cornea_R1{e}']+data[f'Radius_Cornea_R2{e}']).apply(self.Round,args=[0.1,1])

        return data

    def _TransformWavefront(self,data,diopterInterval=0.25,vertexDistance=0.012):
        for e in ['_Left','_Right']:
            for p in [3,5,7]: # for each radius
                data[f'WF_AXIS_R_{p}{e}'] = (data[f'WF_AXIS_R_{p}{e}']
                                             .apply(self.rad2deg)
                                             .apply(self.Round,args=[1,0])
                                             )
                sph = data[f'WF_SPHERE_R_{p}{e}'].copy()
                # Translate sphere to the spectacle plane and round result
                data[f'WF_SPHERE_R_{p}{e}'] = (data[f'WF_SPHERE_R_{p}{e}']
                                                .apply(self.TranslateSphereByVD,args=[vertexDistance])
                                                .apply(self.Round,args=[diopterInterval,2])
                                                )
                data[f'WF_GENERAL_Pd{e}']  = data[f'WF_GENERAL_Pd{e}'].apply(self.Round,delta=0.1,ndigits=1)

                # Translate cylinder to the spectacle plane and round results to the nearest quarter dioptric interval
                data[f'WF_CYLINDER_R_{p}{e}'] = self.TranslateCylinderByVD(sph,data[f'WF_CYLINDER_R_{p}{e}'],vertexDistance).apply(self.Round,delta=diopterInterval)
                for kIdx in ['Photo','Meso']:
                    data[f'WF_RT_Fv_{kIdx}_PupilRadius{e}']   = data[f'WF_RT_Fv_{kIdx}_PupilRadius{e}'].apply(self.Round,delta=0.1,ndigits=2)
                    data[f'WF_RT_Fv_{kIdx}_PupilDiameter{e}'] = 2*data[f'WF_RT_Fv_{kIdx}_PupilRadius{e}']
                    data[f'WF_RT_Fv_{kIdx}_Sphere{e}']        = data[f'WF_RT_Fv_{kIdx}_Sphere{e}'].apply(self.Round,delta = diopterInterval,ndigits=2)
                    data[f'WF_RT_Fv_{kIdx}_Cylinder{e}']      = data[f'WF_RT_Fv_{kIdx}_Cylinder{e}'].apply(self.Round,delta=diopterInterval,ndigits=2)
                    data[f'WF_RT_Fv_{kIdx}_Axis{e}']          = data[f'WF_RT_Fv_{kIdx}_Axis{e}'].apply(self.Round,delta= diopterInterval,ndigits=2)

                # TODO: move to Others
                j0,j45,M = self.ComputePowerVector(data[f'WF_SPHERE_R_{p}{e}'],
                                                   data[f'WF_CYLINDER_R_{p}{e}'],
                                                   data[f'WF_AXIS_R_{p}{e}'])
                data[f'J0_{p}{e}']                  = j0
                data[f'J45_{p}{e}']                 = j45
                data.loc[data.index,f'SphericalEquivalent_{p}{e}'] = M.apply(self.Round,delta=diopterInterval/2,ndigits=3)
                data[f'BlurStrength_{p}{e}']        = ((M**2+j0**2+j45**2)**0.5).apply(self.Round,delta = 0.01, ndigits=2)
        return data

    def _TransformTono(self,data,refractiveIndex=1.3375,iopc_method="doughty_normal"):
        for e in ['_Left','_Right']:
            data[f'IOPcorrected{e}'] = self.IOPc(data[f'Tono_MEASURE_Average{e}'],
                                                 data[f'Pachy_MEASURE_Thickness{e}'],
                                                 data[f'Topo_Sim_K_K1{e}'],
                                                 data[f'Topo_Sim_K_K2{e}'],
                                                 method=iopc_method,
                                                 refractiveIndex=refractiveIndex).apply(self.Round,args=[0.1,1])
        return data

    def _TransformPachy(self,data):
        for e in ['_Right','_Left']:
            data[f'Pachy_MEASURE_IridoAngleR{e}'] = (data[f'Pachy_MEASURE_IridoAngleR{e}']
                                                    .apply(self.Round,args=[1,0])
                                                    )
            data[f'Pachy_MEASURE_IridoAngleL{e}'] = (data[f'Pachy_MEASURE_IridoAngleL{e}']
                                                    .apply(self.Round,args=[1,0,[-1000,np.nan,None,0]])
                                                    )
            data[f'Pachy_MEASURE_Thickness{e}']   = (data[f'Pachy_MEASURE_Thickness{e}']
                                                    .apply(self.Round,args=[0.01,2,[-1000,np.nan,None,0]])
                                                    )
            data[f'Pachy_MEASURE_WhiteToWhite{e}']  = (data[f'Pachy_MEASURE_WhiteToWhite{e}']
                                                        .apply(self.Round, args=[0.01,2,[-1000,np.nan,None,0]])
                                                        )
            data[f'Pachy_MEASURE_KappaAngle{e}']   = (data[f'Pachy_MEASURE_KappaAngle{e}']
                                                    .apply(self.Round,args=[0.1,1])
                                                    )
            # TODO: move to Others
            data[f'AnteriorChamberVol{e}']        = self.AnteriorChamberVolume(data[f'Pachy_MEASURE_Acd{e}'],
                                                                                data[f'Pachy_MEASURE_WhiteToWhite{e}'],
                                                                                data[f'WF_RT_Fv_Meso_PupilRadius{e}'],
                                                                                data[f'Radius_Cornea_Avg{e}']
                                                                                ).apply(self.Round,args=[0.1,1,[-1000,np.nan,None,0]])
        return data

    def _WavefrontAnalysis(self,data):
        """
            Compute Coma, Trefoil, Low order aberrations (LOA)
            and High Order Aberrations (HOA) from the WF Zernike polynomials

            Parameters:
            -----------
                data : DataFrame
                    parsed measurements from the Vx120/130 machine

            Output:
            -------
                data : DataFrame
                    The input DF with added fields for the Coma, Trefoil, angles angles, LOA and HOA
        """
        for eIdx in ['_Right','_Left']:
            # translate from a single Zernike index to double index
            for zIdx in range(36):
                n = int(np.floor(np.sqrt(2*zIdx+1)+0.5)-1) # radial degree
                m = int(2*zIdx-n*(n+2))                    # azimuthal frequency
                for kIdx in ['Photo','Meso']:
                    # Translate the single index to double index
                    data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_Z_{n}_{m}{eIdx}'] = data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_Z_{zIdx}{eIdx}']
                    # Compute the angle theta
                for kIdx in [3,5,7]:
                    data.loc[:,f'WF_ZERNIKE_{kIdx}_Z_{n}_{m}{eIdx}'] = data.loc[:,f'WF_ZERNIKE_{kIdx}_Z_{zIdx}{eIdx}']

            # Compute the angles
            for zIdx in range(36): # up to the number of terms
                n = int(np.floor(np.sqrt(2*zIdx+1)+0.5)-1) # radial degree
                m = int(2*zIdx-n*(n+2))                    # azimothal frequency
                for kIdx in ['Photo','Meso']:
                    if m==0:
                            data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_A_{n}_{abs(m)}{eIdx}'] = 0
                    else:
                        data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_A_{n}_{abs(m)}{eIdx}'] = \
                            np.arctan2(data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_Z_{n}_{-abs(m)}{eIdx}'],data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_Z_{n}_{abs(m)}{eIdx}'])/abs(m)
                            # np.arctan(data[f'WF_RT_Fv_Zernike_{kIdx}_Z_{n}_{-abs(m)}{eIdx}']/data[f'WF_RT_Fv_Zernike_{kIdx}_Z_{n}_{abs(m)}{eIdx}'])/abs(m)
                for kIdx in [3,5,7]:
                    if m ==0:
                        data.loc[:,f'WF_ZERNIKE_{kIdx}_A_{n}_{abs(m)}{eIdx}'] = 0
                    else:
                        data.loc[:,f'WF_ZERNIKE_{kIdx}_A_{n}_{abs(m)}{eIdx}'] = \
                            np.arctan2(data.loc[:,f'WF_ZERNIKE_{kIdx}_Z_{n}_{-abs(m)}{eIdx}'],data.loc[:,f'WF_ZERNIKE_{kIdx}_Z_{n}_{abs(m)}{eIdx}'])/abs(m)
                            # np.arctan(data[f'WF_ZERNIKE_{kIdx}_Z_{n}_{-abs(m)}{eIdx}']/data[f'WF_ZERNIKE_{kIdx}_Z_{n}_{abs(m)}{eIdx}'])/abs(m)

        for eIdx in ['_Right','_Left']:
            for kIdx in ['Photo','Meso']:
                # compute LOA, HOA, Coma, and Trefoil
                # First, compute the angle for each required low-order polynomial
                # LOA is computed using Z4 and Z4 (Z_2^0 and Z_2^2)
                data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_A_9{eIdx}']     = (np.arctan2(data[f'WF_RT_Fv_Zernike_{kIdx}_Z_6{eIdx}'],\
                                                                  data[f'WF_RT_Fv_Zernike_{kIdx}_Z_9{eIdx}'])*180/np.pi)/3
                data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_Coma{eIdx}']    = ((data[f'WF_RT_Fv_Zernike_{kIdx}_Z_7{eIdx}']**2 + \
                                                                  data[f'WF_RT_Fv_Zernike_{kIdx}_Z_8{eIdx}']**2)**0.5).apply(self.Round, args=[0.01,2])
                data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_Trefoil{eIdx}'] = (data[f'WF_RT_Fv_Zernike_{kIdx}_Z_9{eIdx}']/\
                                                                  np.cos(3*data[f'WF_RT_Fv_Zernike_{kIdx}_A_9{eIdx}']*np.pi/180)).apply(self.Round,args=[0.01,2])
                data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_LOA{eIdx}'] = self.Round(np.sum([data[f'WF_RT_Fv_Zernike_{kIdx}_Z_{zIdx}{eIdx}']**2 for zIdx in np.arange(3,6,1)])**0.5,0.01,2)
                data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_HOA{eIdx}'] = self.Round(np.sum([data[f'WF_RT_Fv_Zernike_{kIdx}_Z_{zIdx}{eIdx}']**2 for zIdx in np.arange(6,36,1)])**0.5,0.01,2)
                data.loc[:,f'WF_RT_Fv_Zernike_{kIdx}_Z_12{eIdx}'] = data[f'WF_RT_Fv_Zernike_{kIdx}_Z_12{eIdx}'].apply(self.Round,args=[0.001,3])


        for eIdx in ['_Right','_Left']:
            for kIdx in [3,5,7]:
                # compute LOA, HOA, Coma and Trefoil
                # First, compute the angle for each required low-order polynomial
                data.loc[:,f'WF_ZERNIKE_{kIdx}_A_9{eIdx}']    = (np.arctan2(data[f'WF_ZERNIKE_{kIdx}_Z_6{eIdx}'],\
                                                            data[f'WF_ZERNIKE_{kIdx}_Z_9{eIdx}'])*180/np.pi)/3

                data.loc[:,f'WF_ZERNIKE_{kIdx}_Coma{eIdx}']   = ((data[f'WF_ZERNIKE_{kIdx}_Z_7{eIdx}']**2 + \
                                                            data[f'WF_ZERNIKE_{kIdx}_Z_8{eIdx}']**2)**0.5).apply(self.Round, args=[0.01,2])
                data.loc[:,f'WF_ZERNIKE_{kIdx}_Trefoil{eIdx}'] = (data[f'WF_ZERNIKE_{kIdx}_Z_9{eIdx}']/\
                                                            np.cos(3*data[f'WF_ZERNIKE_{kIdx}_A_9{eIdx}']*np.pi/180)).apply(self.Round,args=[0.01,2])
                data.loc[:,f'WF_ZERNIKE_{kIdx}_LOA{eIdx}'] = self.Round(np.sum([data[f'WF_ZERNIKE_{kIdx}_Z_{zIdx}{eIdx}']**2 for zIdx in np.arange(3,6,1)])**0.5,0.01,2)
                data.loc[:,f'WF_ZERNIKE_{kIdx}_HOA{eIdx}'] = self.Round(np.sum([data[f'WF_ZERNIKE_{kIdx}_Z_{zIdx}{eIdx}']**2 for zIdx in np.arange(6,36,1)])**0.5,0.01,2)

        return data

    def _WavefrontTopoAnalysis(self,data):
        """
         Compute Coma Trefoil, angles HOA and LOA for corenal Zernike using topography data
         Parameters:
         -----------
            data, DataFrame,
                parsed vx120/130 measurements
         Output:
         ----------
            data, DataFrame
                The input DataFrame with added fields for the Coma, Trefoil, HOA, and LOA

        """

        for lIdx in['_Right','_Left']:
            for p in ['3_Photo','5_Meso','7']:
                # First, derive the angles from the Zernike polynomials
                data.loc[data.index,f'Topo_ZERNIKE_{p}_A_0_0{lIdx}'] = 0
                data.loc[data.index,f'Topo_ZERNIKE_{p}_A_1_0{lIdx}'] = 0
                data.loc[data.index,f'Topo_ZERNIKE_{p}_A_2_0{lIdx}'] = 0
                for rIdx in [1,2]:
                    data.loc[data.index,f'Topo_ZERNIKE_{p}_A_{rIdx}_{rIdx}{lIdx}'] = (np.arctan2(data[f'Topo_ZERNIKE_{p}_Z_{rIdx}_-{rIdx}{lIdx}'],\
                                                                        data[f'Topo_ZERNIKE_{p}_Z_{rIdx}_{rIdx}{lIdx}'])*180/np.pi)/rIdx
                # Then, derive the coefficients
                c22 = data[f'Topo_ZERNIKE_{p}_Z_2_2{lIdx}']/np.cos(data[f'Topo_ZERNIKE_{p}_A_2_2{lIdx}']*2*np.pi/180)
                c20 = data[f'Topo_ZERNIKE_{p}_Z_2_0{lIdx}']
                # Use coefficients to compute Total LOA (RMS)
                data.loc[data.index,f'Topo_ZERNIKE_{p}_LOA{lIdx}'] = (np.sqrt(c20**2 + c22**2)).apply(self.Round,delta=0.01,ndigits=2)

                # compute coefficient for Zernike polynomial of high order
                coeffsHOA = []
                for rIdx in range(3,8):
                    if (rIdx%2)==0:
                        # for even polynomials
                        ranks = range(0,rIdx+1,2)
                        for kIdx in ranks:
                            if kIdx ==0:
                                data.loc[data.index,f'Topo_ZERNIKE_{p}_A_{rIdx}_{kIdx}{lIdx}'] = 0
                            else:
                                data.loc[data.index,f'Topo_ZERNIKE_{p}_A_{rIdx}_{kIdx}{lIdx}'] = (np.arctan2(data.loc[:,f'Topo_ZERNIKE_{p}_Z_{rIdx}_-{kIdx}{lIdx}'],\
                                                                                              data.loc[:,f'Topo_ZERNIKE_{p}_Z_{rIdx}_{kIdx}{lIdx}'])*180/np.pi)/kIdx
                            zerkPoly = data[f'Topo_ZERNIKE_{p}_Z_{rIdx}_{kIdx}{lIdx}']
                            coeff    = (zerkPoly/np.cos(kIdx*data[f'Topo_ZERNIKE_{p}_A_{rIdx}_{kIdx}{lIdx}'].values*np.pi/180)).values
                            coeffsHOA.append(coeff)
                            data.loc[data.index,f'Topo_ZERNIKE_{p}_Z_{rIdx}_{kIdx}{lIdx}'] = data[f'Topo_ZERNIKE_{p}_Z_{rIdx}_{kIdx}{lIdx}'].apply(self.Round,delta=0.001,ndigits=3)
                    else:
                        # for odd polynomials
                        ranks = range(1,rIdx+1,2)
                        for kIdx in ranks:
                            data.loc[:,f'Topo_ZERNIKE_{p}_A_{rIdx}_{kIdx}{lIdx}'] = (np.arctan2(data.loc[:,f'Topo_ZERNIKE_{p}_Z_{rIdx}_-{kIdx}{lIdx}'],\
                                                                               data.loc[:,f'Topo_ZERNIKE_{p}_Z_{rIdx}_{kIdx}{lIdx}'])*180/np.pi)/kIdx
                            zerkPoly = data[f'Topo_ZERNIKE_{p}_Z_{rIdx}_{kIdx}{lIdx}']
                            coeff    = (zerkPoly/np.cos(kIdx*data[f'Topo_ZERNIKE_{p}_A_{rIdx}_{kIdx}{lIdx}']*np.pi/180)).values
                            coeffsHOA.append(coeff)
                            data.loc[data.index,f'Topo_ZERNIKE_{p}_Z_{rIdx}_{kIdx}{lIdx}'] = data[f'Topo_ZERNIKE_{p}_Z_{rIdx}_{kIdx}{lIdx}'].apply(self.Round,delta=0.001,ndigits=3)

                    # Compute Total HOA (RMS)
                data.loc[data.index,f'Topo_ZERNIKE_{p}_HOA{lIdx}']     = self.Round(np.sqrt((np.asanyarray(coeffsHOA)**2).sum()),delta=0.01)
                data.loc[data.index,f'Topo_ZERNIKE_{p}_Coma{lIdx}']    = (np.sqrt(data[f'Topo_ZERNIKE_{p}_Z_3_1{lIdx}']**2 + \
                                                                 data[f'Topo_ZERNIKE_{p}_Z_3_-1{lIdx}']**2)).apply(self.Round,delta=0.01,ndigits=2)
                data.loc[data.index,f'Topo_ZERNIKE_{p}_Trefoil{lIdx}'] = (data[f'Topo_ZERNIKE_{p}_Z_3_3{lIdx}']/\
                                                                np.cos(3*data[f'Topo_ZERNIKE_{p}_A_3_3{lIdx}']*np.pi/180)).apply(self.Round,delta=0.01,ndigits=2)

        return data

    @staticmethod
    def _ToID(idStr):
        if isinstance(idStr,str):
            return idStr.lower().replace('-','').replace('/','').replace(' ','')
        else:
            return 'missingID'

    @staticmethod
    def ComputePowerVector(sph,cyl,ax):
        """
         Compute the components of the power vector j0, j45, and the spherical equivalent
         Input to this function can also be pandas.DataFrame and pandas.Series

         Parameters:
         -----------
         sph, float
          sphere component D
         cyl, float
           cylinder component (D)
         ax, float
           axis component (degrees)

         Returns:
         ---------

         j0, float
         j45, flaot,
         M, float, spherical equivalent

        """
        # check if the axis is in radians or degreees
        if np.max(ax)>np.pi:
            # make sure we work in radians
            norm = np.pi/180
        else:
            norm = 1


        j0  = -0.5*cyl*np.cos(2*ax*norm)
        j45 = -0.5*cyl*np.sin(2*ax*norm)
        M   = sph +cyl/2

        return j0,j45,M

    @staticmethod
    def ComputeAge(bDate,eDate):
        """
         compute age (years) at time of examination
         Parameters:
         ---------
         bDate, str
           birthdate
         eDate, str
           examination date
        """
        if pd.notna(bDate) and pd.notna(eDate):
            return (pd.Timestamp(eDate)-pd.Timestamp(bDate)).days/365
        else:
            return np.nan

    @staticmethod
    def _ParseDateDiff(timeDelta,output='years'):
        '''
            translate time delta in days to years
            Parameters
            ---------
            timeDelta, pd.Timedelta
            output, str, default='years'
             output type, years, or days
        '''
        if isinstance(timeDelta,pd.Timedelta):
            if output.lower()=='years':
                return timeDelta.days/365
            elif output.lower()=='days':
                return timeDelta.days
            else:
                ValueError(f'output must be eaither years or days')
        else:
            return np.nan

    @staticmethod
    def AssignGender(sex):
        """
         sex, str
          f= female,
          m= male
         are coded as
          0 = male
          1 = female
        """
        if isinstance(sex,str):
            if sex.lower()=='m':
                return 0
            elif sex.lower()=='f':
                return 1
            else:
                return np.nan
        else:
            return np.nan

    @staticmethod
    def ParseDateToStr(date,dayfirst=True):
        """
         Parameters:
         -----------
          date, str
         Output:
         -------
          date, string
            format: day/month/year
        """
        try:
            eDate = dateParser.parse(date,dayfirst=dayfirst)
            if eDate>datetime.today():
                eDate = eDate.replace(eDate.year-100)
            return f"{eDate.day:02d}/{eDate.month:02d}/{eDate.year}"
        except:
            return np.nan

    @staticmethod
    def Round(value:float,delta:float,ndigits=2,nanValues=[-1000,np.nan,None])->float:
        """
             Round a value to the nearest multiple of delta and truncate
             Check if values are part of the nanValue list and return np.nan is so

             Parameters:
             ----------
             value, float
               the value to round
             delta, float/int/None
              round value to nearest delta
             ndigits, int, default=2
              number of decimal points to truncate to
             nanValues, list, default[-1000,np.nan, None]
              a list of values considered as nan

            Returns:
            --------
             values, float
              Rounded and truncated value
        """
        if delta<0:
            raise ValueError('delta must be positive')

        if value in nanValues or pd.isna(value):
            return np.nan
        else:
            if delta is None:
                return value
            else:
                # round to nearest delta then round and truncate to keep ndigits after the decimal point
                val = round(round((value/delta),ndigits=0)*delta,ndigits=ndigits)
                if ndigits == 0:
                    val = int(val)
                if val<-1000: #  workaround for omitting faulty values (Visionix bug of translating -1000 to angles)
                    val = np.nan
                return val

    @staticmethod
    def AnteriorChamberVolume(acd,wtw,Pr,R):
        """
          Compute an approximation to the anterior chamber volume.
          This computed volume assume a roughly spherical shape of the cornea.
          Parameters:
          -----------
          acd, float
           anterior chamber depth (mm)
          wtw, float
           white-to-white (mm)
          Pr, float
           pupil radius
          R, float
           posterior cornea radius (mm)

          Returns:
          --------
           anterior chamber volume, float
        """
        det     = R**2 - (wtw/2)**2
        # inds    = np.where(det>=0)[0]
        x       = np.where(det>=0,det**0.5,0) # truncate to zero cases in which (wtw/2)>R
        y       = R-acd-x
        # volY    = (np.pi/3)*y*(wtw/2)**2
        volY    = (np.pi/3)*y*((wtw/2)**2+Pr**2 *(wtw*Pr/2))
        vol_acd = (np.pi/3)*(3*(wtw/2)-acd-y)*(acd+y)**2

        return vol_acd-volY

    @staticmethod
    def TranslateSphereByVD(sphere,vd):
        """
            Adjust sphere power to take into account the vertex distance.
            The convention is that the eye is on the origin, and light travels from left to right.
            Therefore, a positive vertext distance translates the power from the cornea to the spectcle plane (forward)
            a negative vertex distance translates the power from the spectacle
            plane to the cornea (backward)

            Parameters:
            -----------
            sphere, float
              sphere power (D)
            vd, float
                vertex distance (mm)

            Returns:
            ----------
            sphre, float,
              sphere power translated by vertex distance

        """
        return sphere/(1+vd*sphere)

    @ staticmethod
    def TranslateCylinderByVD(sphere,cylinder,vd):
        """
          Adjust cylinder power to take into account the vertex distance.

          Parameters:
          ------------
          sphere, float
              sphere power BERFORE translation by vertex distance
          cylinder, float
              cylinder power before translation by vertex distance
          vd, float,
              the vertex distance delta (measured-machine setting)
              the convention is vd<0 from spectacle to corne place
              vd>0 from cornea to spectacle plane
          Output:
          --------
          cylinder, float
             adjusted value by vertex distance
        """
        return (sphere+cylinder)/(1+vd*(sphere+cylinder)) - sphere/(1+vd*sphere)

    def PowerVector2SCA(self,M,J0,J45,vd=0.012):
        """
            Transform the components of the power vector to Sphere Cylinder and Axis

            Parameters:
            ---------
            M, float
             spherical equivalent (D)
            J0, float
             -0.5*C*cos(2*A)
            J45, float
             -0.5*C*Sin(2*A)

            Returns:
            ---------
            (S,C,A) tuple (floats)
             Sphere (D), cylinder (D), Axis (rad)
        """

        theta = 0.5*np.arctan2(J45,J0)

        if J0<0:
            A = theta+0.5*np.pi
        elif J0==0 and J45<0:
            A = 0.75*np.pi
        elif J0==0 and J45>0:
            A = 0.25*np.pi
        elif J0>0 and J45<=0:
            A = theta+np.pi
        elif J0>0 and J45>0:
            A = theta

        C    = -2*(J45**2+J0**2)**0.5
        S    = M-C/2
        Snew = self.TranslateSphereByVD(S,vd)
        Cnew = self.TranslateCylinderByVD(S,C,vd)
        return Snew,Cnew,A

    @staticmethod
    def IOPc(iop,cct,k1,k2,method="doughty_normal",refractiveIndex =1.3375):
        """
         Intraocular pressure corrected
         Parameters:
         -----------
         iop, float
          intraocular pressure (mmHg)
         cct, float
          central cornea thickness (mu m)
         k1,k2, float
          keratometry values (D)
         method, str, default="doughty_normal"
          the method to estimate the corrected IOP by author's name
        """
        if method.lower()=="ehlers":
            return iop - (cct-520)*5/70
        elif method.lower() == "whitacre":
            return iop - (cct-560)*2/100
        elif method.lower()=="doughty_normal":
            return iop- (cct-535)*1.1/50
        elif method.lower()=="doughty_glauc":
            return iop-(cct-535)*2.5/50
        elif method.lower()=="shimmyo":
            return iop+(550-cct)/(18*np.e**(-0.005*iop)) +0.8*(675/(k1 +k2)-7.848837)
        elif method.lower()=="orssengo":
            R  = 0.5*1000*(refractiveIndex-1)*(1/k1 +1/k2)# avg cornea radius
            v  = 0.49
            Tc = 0.52
            Rc = 7.8
            A  = 7.35
            T  = cct/1000
            Bc = 0.6*np.pi*Rc*(Rc-Tc/2)*((1-v**2)**0.5)/Tc**2
            Cc = np.pi*Rc*(Rc-Tc/2)*(1-v)/(A*Tc)
            B  = 0.6*np.pi*Rc*(Rc-T/2)*((1-v**2)**0.5)/T**2
            C  = np.pi*R*(R-T/2)*(1-v)/(A*T)
            return iop*B/(Bc-Cc+C)
        elif method.lower()=="ehlers_table":
            c = np.arange(445,650,10)
            v = [7,6,6,5,4,4,3,2,1,1,0,-1,-1,-2,-3,-4,-4,-5,-6,-6,-8]
            return iop+v[np.argmin(np.abs(cct-c))]
        elif method.lower()=="dresdner_table":
            c = np.arange(475,725,25)
            v = [3.19,2.13,1.07,0.02,-1.04,-2.1,-3.16,-4.21,-5.27,-6.33]
            return iop+v[np.argmin(np.abs(cct-c))]
        elif method.lower()=="herndon_table":
            c = np.arange(405,725,20)
            v = np.arange(7,-9,-1)
            return iop+v[np.argmin(np.abs(cct-c))]
        else:
            # TODO: add error output to logger
            print(f"[Error][vx120Transformer] Unsupported IOPc method {method}")
            return np.nan

    @staticmethod
    def deg2rad(deg):
        """Degrees to Radians"""
        return deg*np.pi/180 if isinstance(deg,(float,int)) else np.nan

    @staticmethod
    def rad2deg(rad):
        """Radians to degrees"""
        return rad*180/np.pi if isinstance(rad,(float,int)) else np.nan

    @staticmethod
    def mm_D(mm,refractiveIndex=1.3375):
        '''
         mm to diopter
         using the refractive index of the medium (default=1.3375)
        '''
        return (refractiveIndex-1)*1000/mm if isinstance(mm,(float,int)) and mm!=0 else np.nan

    @staticmethod
    def _formatDF(val):
        """Format a DataFrame"""
        if isinstance(val,float):
            if np.isnan(val):
                return '--'
            else:
                return f'{val:.2f}'
        else:
            return str(val)