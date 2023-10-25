import os
from shutil import Error
import xml.etree.ElementTree as ET
from dateutil import parser as dateParser
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors._typedefs
import sklearn.neighbors._quad_tree
import sklearn.ensemble._forest
import sklearn.utils._cython_blas
import sklearn.tree._utils
import sklearn.utils._weight_vector
import scipy.spatial.transform._rotation_groups
import json
# from autorefeyelib.Refraction import vx120Imputer
from autorefeyelib.Refraction import vx120Transformer

class Predictor:
    """
     A subjective refraction predictor class, comprised of Cylinder and Sphere trained classifiers.
     This class is meant to be used with a transformed vx120 data (See vx120Transformer class)
     It accepts measuremets in a DataFrame format and returns a dataframe with predicted results and
     their confidence based on inclusion criteria of the trained model
     The trained models must be included in a model folder along with refraction_params.json, which
     includes a dictioinary of inclusion criteria.
     Prediction for subjective refraction is also translated to contact lenses

     The predictor class can export prediction in the format of a vx40 (Visionix) xml file

    """
    # TODO: add re-training loop

    def __init__(self):
        # Load VX db sphere and cylinder classifiers
        self._cylThreshLow        = -0.5  #TODO: move to parameters
        self._cylThreshHigh       = -0.75 #TODO: move to parameters
        self._sphereModelLoaded   = False
        self._cylinderModelLoaded = False
        self._vxDataLoaded        = False
        self._transformer         = vx120Transformer.Transformer()
        self._CheckForFileExistance()
        self._vx40_xml            = ET.parse(os.path.join(os.path.dirname(__file__),'etc','VX40_Sample.xml'))
        classifierPath            = os.path.join(os.path.dirname(__file__),'models')
        self._LoadCylinderClassifier(os.path.join(classifierPath,'CylModel.sav'))
        self._LoadSphereClassifier(os.path.join(classifierPath,'SphModel.sav'))
        model_params            = json.load(open(os.path.join(classifierPath,"refraction_params.json"),'r'))
        if 'inclusion_criteria' in model_params.keys():
            self.inclusion_criteria = model_params['inclusion_criteria']
        else:
            raise ValueError("Missing inclusion criteria from model parameters. Please make sure refraction_params.json is located in the model folders and includes the key inclusion_criteria (dict)")

    def _CheckForFileExistance(self):
        """
         Check if needed files are present.
         Needed files: SphModle.sav, CylModel.sav, refraction_params.json
        """
        # check if the model folder exist
        modelFolderPath  = os.path.join(os.path.dirname(__file__),'models')
        if not os.path.exists(modelFolderPath):
            raise FileExistsError("Models folder does not exist. Make sure the project folder contains models folder with sph and cyl classifiers")
        else:
            if not os.path.exists(os.path.join(modelFolderPath,"CylModel.sav")):
                raise FileExistsError(f"Cylinder classifier file was not found in path. Expected in {os.path.join(modelFolderPath,'CylModel.sav')}")
            if not os.path.exists(os.path.join(modelFolderPath,"CylModel.sav")):
                raise FileExistsError(f"Sphere classifier file not was found in path. Expected in {os.path.join(modelFolderPath,'SphModel.sav')}")
            if not os.path.exists(os.path.join(modelFolderPath,"refraction_params.json")):
                raise FileExistsError(f"Subjective refraction classifier parameter file was not found. Expected in {os.path.join(modelFolderPath,'refraction_params.json')}")

    def _LoadCylinderClassifier(self,modelPath:str):

        with open(modelPath, 'rb') as file:
            self.cylModel             = joblib.load(file)
            self._cylinderModelLoaded = True

    def _LoadSphereClassifier(self,modelPath:str):

        with open(modelPath, 'rb') as file:
            self.sphModel           = joblib.load(file)
            self._sphereModelLoaded = True

    def PredictSubjectiveRefraction(self,measurements:pd.DataFrame, vx40Data,returnVals='all',vertexDistance=0.012, pred_vx40_diff_thresh_sph=0.75,pred_vx40_diff_thresh_cyl=0.5)->pd.DataFrame:
        """
           Predict subjective sphere and cylinder

           Parameters:
           -----------
            measurements : pandas.DataFrame
                a datframe including patient measurements as parsed from the vx parser using raw data
                Make sure the data has gone through Transform in the vx120Transformer prior to use
            returnVals : str, default='all'
                the output type, either all, to return all computed and transformed values
                or 'predicted' to return only predicted values
            vx40Data : pandas.DataFrame
                parsed vx40 data (see xmlParser for structure)

            Returns
            -------
            prediction : pandas.DataFrame
                a dataframe including all vx120 measurements and added predicted sphere cylinder and axis components
                a confidence field is added to indicate the streength of the prediciton within the inclusion ranges of training features
        """

        print(f"[INFO][VXpredictor] Predicting subjective sphere and cylinder")
        # Make sure the data is imputed and transformed
        self._CheckInputDataIntegrity(measurements)
        # computed needed extra features for prediction
        self._ComputeAddedFeatures(measurements)

        # Get left and right data for predicting sphere (strip _Left, _Right from column names)
        # complete missing fields and impute missing values
        featuresSph = self.sphModel.featuresUsed
        measurementsSph_Left,measurementsSph_Right = self._PreparePatientMeasurements(measurements,featuresSph)

        # Get left and right data for predicting cylinder (strip _Left, _Right from column names)
        featuresCyl = self.cylModel.featuresUsed
        measurementsCyl_Left,measurementsCyl_Right = self._PreparePatientMeasurements(measurements,featuresCyl)
        prediction = measurements.copy()
        # Add predicted values to current patient measurements
        prediction['Predicted_Sphere_Left']    = measurements['WF_SPHERE_R_3_Left']    - self._PredictSphereDelta(measurementsSph_Left)
        prediction['Predicted_Sphere_Right']   = measurements['WF_SPHERE_R_3_Right']   - self._PredictSphereDelta(measurementsSph_Right)
        prediction['Predicted_Cylinder_Left']  = measurements['WF_CYLINDER_R_3_Left']  - self._PredictCylinderDelta(measurementsCyl_Left)
        prediction['Predicted_Cylinder_Right'] = measurements['WF_CYLINDER_R_3_Right'] - self._PredictCylinderDelta(measurementsCyl_Right)

        if prediction['Predicted_Sphere_Left'].isna().any()==False:
            prediction['Predicted_Add_Left']   = prediction['Age'].apply(self._ComputeAdd)
            prediction['Predicted_Axis_Left']  = prediction['WF_AXIS_R_3_Left']
        else:
            prediction['Predicted_Add_Left']   = np.nan
            prediction['Predicted_Axis_Left']  = np.nan
        if prediction['Predicted_Sphere_Right'].isna().any()==False:
            prediction['Predicted_Axis_Right'] = prediction['WF_AXIS_R_3_Right']
            prediction['Predicted_Add_Right']  = prediction['Age'].apply(self._ComputeAdd)
        else:
            prediction['Predicted_Axis_Right'] = np.nan
            prediction['Predicted_Add_Right']  = np.nan

        # Translate prediction for prescription glasses to contact lenses prescription using vertex distance 0
        prediction = self._PredictContactLenses(prediction,
                                                 cylThreshLow=self._cylThreshLow,
                                                 cylThreshHigh=self._cylThreshHigh,
                                                 vertexDistance=vertexDistance)
        if vx40Data is not None:
            spectacle_sph = (vx40Data['optometry_LSM_mesurement_measure_REF_ref_right_sphere'].values[0],
                            vx40Data['optometry_LSM_mesurement_measure_REF_ref_left_sphere'].values[0])
            spectacle_cyl = (vx40Data['optometry_LSM_mesurement_measure_REF_ref_right_sphere'].values[0],
                            vx40Data['optometry_LSM_mesurement_measure_REF_ref_left_sphere'].values[0])
        else:
            spectacle_sph = (None, None)
            spectacle_cyl = (None, None)

        # Assign confidence to predictions
        self._AssignConfidenceToPrediction(measurements,prediction,spectacle_sph,spectacle_cyl,
                                        thresh_sph=pred_vx40_diff_thresh_sph,
                                        thresh_cyl=pred_vx40_diff_thresh_cyl)

        # Convert to obtain the right dtypes
        prediction = prediction.astype({'Predicted_Sphere_Left':float,
                                        'Predicted_Sphere_Right':float,
                                        'Predicted_Cylinder_Left':float,
                                        'Predicted_Cylinder_Right':float,
                                        'Predicted_Axis_Left':int,
                                        'Predicted_Axis_Right':int,
                                        'Predicted_Contact_Axis_Left':int,
                                        'Predicted_Contact_Axis_Right':int,
                                        'Predicted_Contact_Sphere_Left':float,
                                        'Predicted_Contact_Cylinder_Left':float,
                                        'Predicted_Contact_Sphere_Right':float,
                                        'Predicted_Contact_Cylinder_Right':float},
                                        errors='ignore')

        if returnVals=='predicted':
            dCols = []
            for c in prediction.columns:
                if c.find('Predicted')==-1:
                    dCols.append(c)
            prediction.drop(columns=dCols,inplace=True)

        return prediction

    def _AssignConfidenceToPrediction(self,features, prediction, spectacle_sph, spectacle_cyl,thresh_sph=0.75,thresh_cyl=0.5):
        """
         Assign confidence to prediction based on inclusion criteria
         and the difference between the predicted values and the corresponding
         values of the current spectacle

         Parameters
         ----------
         features   : pandas.DataFrame
            parsed vx120 measurements
         prediction : pandas.DataFrame
           prediction dataframe, with populated Predicted_Sphere  and Predicted_Cylinder for Left and Right
         spectacle_sph : list, np.ndarray, tuple
           2 element float of [sphere right, sphere left]
         spectacle_cyl : list, np.ndarray(2), tuple
           2 element float of [cylinder right, cylinder left]
         thresh_sph : float
           threshold for the difference between predicted sphere and current specatcle sphere, above which the indicator is set to True

         thresh_cyl : float
           threshold for the difference between predicted cylinder and current specatcle cylinder, above which the indicator is set to True

         Returns
         -------
          prediction : pd.DataFrame
           the input prediction dataFrame with added two fields for each eye
            Prediction_Confidence_Right and Left, which tells how many input features are within the inclusion range while training
            and Prediction_diff_spectacle_Sphere and _Cylinder for _Left and _Right
            which indicates True in the case the differece is bigger than the input
            and False, otherwise.
        """
        if len(spectacle_sph)!=2:
            raise ValueError(f'spectacle_sph must be two element tuple list or array of float, got length = {len(spectacle_sph)}')
        if len(spectacle_cyl)!=2:
            raise ValueError(f'spectacle_cyk must be two element tuple list or array of float, got length = {len(spectacle_cyl)}')

        # compute the confidence of prediction on the current data
        validRight, validLeft = self._CheckInputMeasurmentRanges(features,self.inclusion_criteria)
        # Compute the confidence of measurements based on inclusion_criteria
        pred_confidence_right  = sum(validRight.values())/len(validRight.keys())
        pred_confidence_left   = sum(validLeft.values())/len(validLeft.keys())
        prediction['Predicted_Confidence_Right'] = pred_confidence_right
        prediction['Predicted_Confidence_Left']  = pred_confidence_left

        # right sphere
        if not pd.isnull(spectacle_sph[0]):
            sph_diff = prediction['Predicted_Sphere_Right'].values[0]-spectacle_sph[0]
            if abs(sph_diff)>thresh_sph:
                print(f'[Info][SubjRef] Right eye predicted sphere is signigficantly different than current glasses, with absolute difference of {sph_diff}')
                prediction['Predicted_diff_spectacles_Sphere_Right'] = True
            else:
                prediction['Predicted_diff_spectacles_Sphere_Right'] = False
        else:
            prediction['Predicted_diff_spectacles_Sphere_Right'] = False

        # Left sphere
        if not pd.isnull(spectacle_sph[1]):
            sph_diff = prediction['Predicted_Sphere_Left'].values[0]-spectacle_sph[1]
            if abs(sph_diff)>thresh_sph:
                print(f'[Info][SubjRef] Left eye predicted sphere is signigficantly different than current glasses, with absolute difference of {sph_diff}')
                prediction['Predicted_diff_spectacles_Sphere_Left'] = True
            else:
                prediction['Predicted_diff_spectacles_Sphere_Left'] = False
        else:
            prediction['Predicted_diff_spectacles_Sphere_Left'] = False

        # right cylinder
        if not pd.isnull(spectacle_cyl[0]):
            cyl_diff = prediction['Predicted_Cylinder_Left'].values[0]-spectacle_cyl[0]
            if abs(cyl_diff)>thresh_cyl:
                print(f'[Info][SubjRef] Right eye predicted cylinder is signigficantly different than current glasses, with absolute difference of {cyl_diff}D')
                prediction['Predicted_diff_spectacles_Cylinder_Right'] = True
            else:
                prediction['Predicted_diff_spectacles_Cylinder_Right'] = False
        else:
            prediction['Predicted_diff_spectacles_Cylinder_Right'] = False

        # left cylinder
        if not pd.isnull(spectacle_cyl[1]):
            cyl_diff = prediction['Predicted_Cylinder_Right'].values[0]-spectacle_cyl[1]
            if abs(cyl_diff)>thresh_cyl:
                print(f'[Info][SubjRef] Left eye predicted cylinder is signigficantly different than current glasses, with absolute difference of {cyl_diff}D')
                prediction['Predicted_diff_spectacles_Cylinder_Left'] = True
            else:
                prediction['Predicted_diff_spectacles_Cylinder_Left'] = False
        else:
            prediction['Predicted_diff_spectacles_Cylinder_Left'] = False

        return prediction

    def _CheckInputDataIntegrity(self,measurements:pd.DataFrame)->pd.DataFrame:
        """
            Check if the input measurements are valid, impute missing or invalid values and transform
            Parameters:
            -----------
            measurements, DataFrame
             parsed vx120 measurements
            Output:
            -------
             measurements, DataFrame
              measurements after completing missing values and adjusting for output
        """
        # measurements = self._imputer.ImputeDF(measurements.copy())
        if '_isTransformed' in measurements.columns:
            if (measurements['_isTransformed'].isna().any()==True) or (measurements['_isTransformed']==False).all():
                print('[Info][vx120Predictor] Transforming data')
                measurements = self._transformer.Transform(measurements)
        else:
            print('[Info][vx120Predictor] Transforming data')
            measurements = self._transformer.Transform(measurements)

        return measurements

    def _ComputeAddedFeatures(self,data:pd.DataFrame)->pd.DataFrame:
        """
            Compute additional features based on objective measurements of the vx120
            computed cvalues are:
            * acd to pupil radius ratio
            * pupil radius to wtw ratio
            * kRatio, K2/K1 ratio between keratometry on principle meridians

            Parameters:
            -----------
            data, DataFrame
             patient data after data transform (see Transform method)

            Returns:
            ---------
            data, pandas.DataFrame
              data with appended fields of computed added features
        """

        for eIdx in ['_Left','_Right']:
            # ACD to pupil radius
            data[f'AcdToPupilRadius_Ratio{eIdx}'] = 0.5*data[f'Pachy_MEASURE_Acd{eIdx}']\
                                                            /data[f'WF_RT_Fv_Meso_PupilRadius{eIdx}']
            # pupil radius to wtw ratio
            data[f'PupilRadiusToW2W_Ratio{eIdx}'] = 2*data[f'WF_RT_Fv_Meso_PupilRadius{eIdx}']\
                                                        /data[f'Pachy_MEASURE_WhiteToWhite{eIdx}']
            data[f'kRatio{eIdx}']            = data[f'Topo_Sim_K_K1{eIdx}']/data[f'Topo_Sim_K_K2{eIdx}']


        return data

    @staticmethod
    def _ComputeAdd(age:float)->float:
        """
          Compute the Addition component of refraction according to the age
          Parameters:
          ---------
           age, float

        """
        if age>=42 and age<45:
            add = 1
        elif age>=45 and age<51:
            add = 1.5
        elif age>=51 and age<56:
            add = 1.75
        elif age>=56 and age <=110:
            add = 2.00
        else:
            add = 0
        return add

    def _PredictContactLenses(self,prediction:pd.DataFrame,cylThreshLow=-0.5,cylThreshHigh=-0.75,vertexDistance=0.012,diopter_interval=0.5)->pd.DataFrame:
        """
             Translate the subjective prediction of glasses to contact lenses
             by preforming a reverse transform of the power using the negative vertex distance

             Parameters:
             -----------
             prediction, DataFrame
                a dataframe with all patient measurements, including predicted refraction fields:
                Predicted_Sphere_Left,Predicted_Sphere_Right
                Predicted_Cylinder_Left,Predicted_Cylinder_Right
            cylThresh, float
                a threshold above which the cylinder is zerod out and the sphere is computed as the spherical equivalent
            vertexDistance, float, default=0.012m
                the vertex distance from cornea aterior apex to the spectacle plane
            diopter_interval, float, default=0.5
              the dipter to which to round translated powers

            Output:
            --------
            prediction, DataFrame
                DataFrame with patient measurements, including the new fields:
                Predicted_Sphere_Contact_Left,Predicted_Sphere_Contact_Right
                Predicted_Cylinder_Contact_Left, Predicted_Cylinder_Contact_Right
        """
        ind       = prediction.index[0]
        sph_Left  = prediction.loc[ind,'Predicted_Sphere_Left']
        sph_Right = prediction.loc[ind,'Predicted_Sphere_Right']
        cyl_Left  = prediction.loc[ind,'Predicted_Cylinder_Left']
        cyl_Right = prediction.loc[ind,'Predicted_Cylinder_Right']
        ax_Left   = prediction.loc[ind,'Predicted_Axis_Left']
        ax_Right  = prediction.loc[ind,'Predicted_Axis_Right']

        if cyl_Left>=cylThreshLow:
            sph_Left = sph_Left+cyl_Left/2 # assign the spherical equivalent
            cyl_Left = 0                   # zero out the cylinder
            ax_Left  = 0
        elif cyl_Left>=cylThreshHigh and cyl_Left<cylThreshLow:
            if abs(cyl_Left)<abs(sph_Left):
                sph_Left = sph_Left+cyl_Left/2 # assign the spherical equivalent
                cyl_Left = 0
                ax_Left  = 0

        if cyl_Right>=cylThreshLow:
            sph_Right = sph_Right+cyl_Right/2 # assign the spherical equivalent
            cyl_Right = 0                     # zero out the cylinder
            ax_Right  = 0
        elif cyl_Right>=cylThreshHigh:
            if abs(cyl_Right)<abs(sph_Right):
                sph_Right = sph_Right+cyl_Right/2 # assign the spherical equivalent
                cyl_Right = 0
                ax_Right  = 0

        # translate to left and right contact lenses (taking the negative vertex distance for reverse transform)
        prediction.loc[ind,'Predicted_Contact_Sphere_Left']    = self._transformer.Round(self._transformer.TranslateSphereByVD(sph_Left,-vertexDistance),diopter_interval,2)
        prediction.loc[ind,'Predicted_Contact_Cylinder_Left']  = self._transformer.Round(self._transformer.TranslateCylinderByVD(sph_Left,cyl_Left,-vertexDistance),diopter_interval,2)
        prediction.loc[ind,'Predicted_Contact_Sphere_Right']   = self._transformer.Round(self._transformer.TranslateSphereByVD(sph_Right,-vertexDistance),diopter_interval,2)
        prediction.loc[ind,'Predicted_Contact_Cylinder_Right'] = self._transformer.Round(self._transformer.TranslateCylinderByVD(sph_Right,cyl_Right,-vertexDistance),diopter_interval,2)
        prediction.loc[ind,'Predicted_Contact_Axis_Left']      = self._transformer.Round(ax_Left,1,0)
        prediction.loc[ind,'Predicted_Contact_Axis_Right']     = self._transformer.Round(ax_Right,1,0)

        return prediction

    def _PreparePatientMeasurements(self,measurements:pd.DataFrame,features:list)->tuple:
        """
         Organize measurements vector of the vx120/130 into right and left eye data
         to use in classifiers. This function strips the _Left or _Right suffix from the feature
         names and prepares two DataFrame structure to be used for the Sphere Or Cylinder classifiers
         by the feature names listed in the feature variable

         Parameters:
         -----------
         measuremtnts: DataFrame
             a dataframe row from parsed vx data
         features: list
            a list of features to use. Numes of features must be similar to column names in the vxDB
         imputeMissing: bool
            complete missing data in feature vectors using valid values in the vxDB

         Returns:
         ---------
         measurementLeft, measurementRight, tuple(DataFrame)
          measuremtns split into right and left eyes, keeping field names
          stripping the Left, Right suffix in the field names
        """
        ind               = measurements.index.values
        measurementsRight = pd.DataFrame(index = ind)
        measurementsLeft  = pd.DataFrame(index = ind)
        for fIdx in features:
            # for left eye features
            if fIdx+'_Left' in measurements.keys():
                val = float(measurements[fIdx+'_Left'])
                measurementsLeft[fIdx] = val

            # For right eye features
            if fIdx+'_Right' in measurements.keys():
                val  = float(measurements[fIdx+'_Right'])
                measurementsRight[fIdx] = val

            # for non left or right information (e.g. age/name)
            if fIdx in measurements.keys():
                measurementsLeft[fIdx]  = np.asanyarray(measurements[fIdx])
                measurementsRight[fIdx] = np.asanyarray(measurements[fIdx])

        return measurementsLeft, measurementsRight

    def _PredictSphereDelta(self,features:pd.DataFrame)->float:
        """
         Predict the sphere correction based on feature vector
         Parameters:
         -----------
         features: DataFrame or Series
         Output:
         -------
         sphDelta, float
            predicted sphere delta (objective-subjective) in the spectacle plane
        """
        if isinstance(features,pd.Series):
            features = features.to_frame()

        if self._sphereModelLoaded:
            # pat_data = np.asanyarray(features).reshape(1,-1)
            if np.isfinite(features[self.sphModel.feature_names_in_].values).all():
                sphDelta = self.sphModel.sphereDelta[self.sphModel.predict(features)[0].astype(int)]
            else:
                sphDelta = np.nan
            return sphDelta
        else:
            raise(Exception('[Error][VXpredictor] sphere model was not loaded'))

    def _PredictCylinderDelta(self,features:pd.DataFrame)->float:
        """
            Predict subjective cylinder based on feature vector
            Parameters:
            ----------
             features, DataFrame/Series
                feature vector. Feature names as the keys of the DataFrame must correspond to the feture
                expected by the model
            Output:
            -------
             cylDelta, float
              predicted cylinder delta (objective-subjective)

        """
        if isinstance(features,pd.Series):
            features = features.to_frame()

        if self._cylinderModelLoaded:
            # pat_data = np.asanyarray(features).reshape(1,-1)
            if np.isfinite(features[self.cylModel.feature_names_in_].values).all():
                cylDelta  = self.cylModel.cylinderDelta[self.cylModel.predict(features)[0].astype(int)]
            else:
                cylDelta = np.nan
            return cylDelta
        else:
            raise(Exception('[Error][VXpredictor] cylinder model was not loaded'))

    def _CheckInputMeasurmentRanges(self,measurements:pd.DataFrame, inclusion_criteria:dict)->tuple:
        """
         Check if input measurements are within the inclusion ranges

         Parameters:
         ------
         measurements, pandas.DataFrame
          parsed and transformed vx120 measurements

         Returns:
         --------
         validRight/validLeft, dict
            indicators wheather or not the data is within the specified ranges, according to the keys\
            of the dictionary inclusino_criteria
        """
        # verufy input variables
        if not isinstance(measurements,(pd.DataFrame,pd.Series)):
            raise TypeError(" input variable measurements must be of type pandas.DataFrame, got {measurements.__class__}")
        if not isinstance(inclusion_criteria,dict):
            raise TypeError("input variable inclusion_criteria must be a of type dictionary, got {inclusion_criteria.__class__}")

        validRight = {}
        validLeft  = {}
        for cIdx in inclusion_criteria.keys():
            # for left
            m_keys = measurements.keys()
            if cIdx+'_Left' in m_keys:
            # for right
                if (measurements[cIdx+'_Left'].values<=np.max(inclusion_criteria[cIdx])) &\
                    (measurements[cIdx+'_Left'].values>=np.min(inclusion_criteria[cIdx])):
                    validLeft[cIdx] = True
                else:
                    validLeft[cIdx] = False
            if cIdx+'_Right' in m_keys:
            # for non left or right feature
                if (measurements[cIdx+'_Right'].values<=np.max(inclusion_criteria[cIdx])) &\
                        (measurements[cIdx+'_Right'].values>=np.min(inclusion_criteria[cIdx])):
                    validRight[cIdx] = True
                else:
                    validRight[cIdx] = False
            if cIdx in m_keys:
                if (measurements[cIdx].values<=np.max(inclusion_criteria[cIdx])) &\
                        (measurements[cIdx].values>=np.min(inclusion_criteria[cIdx])):
                    validRight[cIdx] = True
                    validLeft[cIdx]  = True
                else:
                    validRight[cIdx] = False
                    validLeft[cIdx]  = False
        return validRight, validLeft

    def ExportPredictionToVx40xml(self,outputFolder:str,prediction:pd.DataFrame):
        """
            Generate a  vx40 xml with predicted refraction and contact lenses values

            Parameters:
            -----------
            prediction, DataFrame
                a dataframe with columns: ID, FirstName, surname, BirthDate, Sphere_Right/_Left, Cylinder_right/_Left, Axis_Right/_Left
        """
        def _setxmlValue(root,nPath,val):
            """
             A utility function to populate xml fields
             Parameters:
             -----------
              root, xml root
              nPath, list, nested field header names
              val, value to assign

            """
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
                    k.text = val
            return root

        # Insert under <subbjective measurements> <LSM_measurements> <measure> <sphere> <cylinder> <axis>
        # to be able to pass to the Oplus
        root = self._vx40_xml.getroot()
        n    = dateParser.parse('010100').now()
        # lsm measurement
        index = prediction.index[0]
        _setxmlValue(root,['optometry','company'],"Mikajaki")
        _setxmlValue(root,['optometry','model_name'],"AutoRef")
        _setxmlValue(root,['optometry','date'],f"{n.day:02d}/{n.month:02d}/{n.year}")
        _setxmlValue(root,['optometry','time'],f"{n.hour:02d}:{n.minute:02d}")
        _setxmlValue(root,['optometry','patient','ID'],str(prediction['ID'][index]))
        _setxmlValue(root,['optometry','patient','first_name'],str(prediction['Firstname'][index]))
        _setxmlValue(root,['optometry','patient','last_name'],str(prediction['Surname'][index]))
        _setxmlValue(root,['optometry','patient','gender'],"m" if prediction['Gender'][index]==0.0 else "f")
        _setxmlValue(root,['optometry','patient','birthday'],str(prediction['BirthDate'][index]))
        # NOTE: the typo appears in the original visionix xml,
        # and is not corrected to allow EMR systems to correctly parse such files
        _setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_right','sphere'],\
                        f"{prediction['Predicted_Sphere_Right'][index]}")
        _setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_right','cylinder'],\
                        f"{prediction['Predicted_Cylinder_Right'][index]}")
        _setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_right','axis'],\
                        f"{prediction['Predicted_Axis_Right'][index]}")
        _setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_right','addition'],\
                        f"{prediction['Predicted_Add_Right'][index]}")

        _setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_left','sphere'],\
                        f"{prediction['Predicted_Sphere_Left'][index]}")
        _setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_left','cylinder'],\
                        f"{prediction['Predicted_Cylinder_Left'][index]}")
        _setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_left','axis'],\
                        f"{prediction['Predicted_Axis_Left'][index]}")
        _setxmlValue(root,['optometry','LSM_mesurement','measure_REF','ref_left','addition'],\
                        f"{prediction['Predicted_Add_Left'][index]}")

        # contact lenses
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','sphere'],\
                        f"{prediction['Predicted_Contact_Sphere_Right'][index]}")
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','cylinder'],\
                        f"{prediction['Predicted_Contact_Cylinder_Right'][index]}")
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','axis'],\
                        f"{prediction['Predicted_Contact_Axis_Right'][index]}")
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','manufacturer'],'Johnson & Johnson')
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','model'],'1-Day Acuvue Moist 180')
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','diameter'],'14.2')
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','right','base_curve'],'8.5')

        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','sphere'],\
                        f"{prediction['Predicted_Contact_Sphere_Left'][index]}")
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','cylinder'],\
                        f"{prediction['Predicted_Contact_Cylinder_Left'][index]}")
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','axis'],
                        f"{prediction['Predicted_Contact_Axis_Left'][index]}")
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','model'],'1-Day Acuvue Moist 180')
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','diameter'],'14.2')
        _setxmlValue(root,['optometry','subjective_mesurement','contact_lens','left','base_curve'],'8.5')

        xmlName = str(prediction['ID'][index]).replace('/','_').replace(' ','')+"_predicted.xml"
        # with open(os.path.join(outputFolder,xmlName),'w') as xmlFile:
        destination = os.path.join(outputFolder, xmlName)
        self._vx40_xml.write(destination, encoding='utf-8')
        return destination

    @staticmethod
    def FormatPredictionAsString(prediction:pd.DataFrame,rtype="spectacles")->str:
        """
          Format the predicted refraction as a string

         Parameters:
         -----------
         prediction, DataFrame
          predicted values
         rtype, str, default="spectacles"
           options: spectacles, contacts
        """
        if len(prediction)!=0:
            cp = prediction
            ds = u'\N{DEGREE SIGN}'
            ind = prediction.index[0]
            if rtype=="spectacles":
                refStr = f"{cp['Predicted_Sphere_Right'][ind]:.2f}"\
                       + f"({cp['Predicted_Cylinder_Right'][ind]:.2f})"\
                       + f"{cp['Predicted_Axis_Right'][ind]:.0f}"+ds \
                       +" / " \
                       + f"{cp['Predicted_Sphere_Left'][ind]:.2f}"\
                       + f"({cp['Predicted_Cylinder_Left'][ind]:.2f})"\
                       + f"{cp['Predicted_Axis_Left'][ind]:.0f}"+ds \
                       + f" *A {cp['Predicted_Add_Left'][ind]:.2f}"
            elif rtype =="contacts":
                refStr = f"{cp['Predicted_Contact_Sphere_Right'][ind]:.2f}"\
                       + f"({cp['Predicted_Contact_Cylinder_Right'][ind]:.2f})"\
                       + f"{cp['Predicted_Contact_Axis_Right'][ind]:.0f}"+ds \
                       + " / " \
                       + f"{cp['Predicted_Contact_Sphere_Left'][ind]:.2f}"\
                       + f"({cp['Predicted_Contact_Cylinder_Left'][ind]:.2f})"\
                       + f"{cp['Predicted_Contact_Axis_Left'][ind]:.0f}"+ds
            else:
                print(f"[Warn] unknown option type={rtype}")
                refStr = ""
            return refStr
        else:
            return ""


