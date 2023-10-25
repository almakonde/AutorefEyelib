#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Jan 28 15:12:02 2020
    @description: This class is a framework to run the Random Forest (RF) classifier in the task of classification of
    vx130 measurements. This class is able to form three differnt predictions:
    1. predict the spherical equivalent delta set: |Delta|<=delta and |Delta|>delta sets, where Delta is a threshold of the spherical equivalent delta.
    2. predict the delta between the objective and subjective sphere and cylinder, and
    3. directly predict the subjective sphere and cylinder based on objective vx measurements.
    @author: ofir shukron
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn import preprocessing
import sklearn.feature_selection
from scipy import stats
from scipy.ndimage import median_filter
import itertools
import pickle
# import matplotlib
from autorefeyelib.Refraction import vx120Transformer
from autorefeyelib.Refraction import vx120Imputer

class Classifier:
    """
        A framework for the classification of  VX -EMR combined data
        into a set with |Delta|<=delta using several test classifiers
        where Delta is  either the spherical equivalent, sphere or cylinder
        delta between objective and subjective refraction
     """
    def __init__(self):
        print("INITIALIZING")
        self.data           = pd.DataFrame()
        self.Left           = pd.DataFrame()
        self.Right          = pd.DataFrame()
        self.dataParsed     = False
        self.dataLoaded     = False
        self.sphModel       = None  # sphere clasifier
        self.cylModel       = None  # cylinder classifier
        self.vertexDistance = -0.012
        self.transformer    = vx120Transformer.Transformer()
        # features below are for either left or right eye, where an underscore _Left, or _Right, appears in the data set.
        # some featues are computed directly from the parser vxParser.
        # All featues are from the VX120/130. The subjective refraction is from the EMR.
        # featureList is a dictionary with keys as feature names and values as their ranges
        # TODO: move features and ranges to xml or json file
        self.featureList    = {
                               'Age':[15,100],
                               'Gender':[],
                               'Topo_Sim_K_K1':[],                           # K1
                               'Topo_Sim_K_K2':[],                           # K2
                               'WF_SPHERE_R_3':[-5,6],                       # sphere
                               'WF_CYLINDER_R_3':[-6,0],                     # cylinder
                               'WF_AXIS_R_3':[],                             # Axis (radians)
                               'WF_RT_Fv_Zernike_Photo_di_Z_2_-2':[-1,1],    # astigmatism
                               'WF_RT_Fv_Zernike_Photo_di_Z_4_0':[-0.2,0.2], # spherical aberration
                               'WF_RT_Fv_Zernike_Photo_di_Z_3_1':[-0.3,0.3], # primary coma photo
                               'WF_RT_Fv_Meso_PupilRadius':[],               # pupil radius
                               'WF_RT_Fv_Zernike_Meso_di_Z_2_-2':[-2,2],     # astigmatism
                               'WF_RT_Fv_Zernike_Meso_di_Z_4_0':[-0.3, 0.3], # spherical abberation diopter
                               'WF_RT_Fv_Zernike_Meso_di_Z_3_1':[-0.4,0.4],  # primary coma meso
                               'Pachy_MEASURE_Acd':[1,5],                    # anterior chamber depth
                               'Pachy_MEASURE_WhiteToWhite':[8,16],          # white to white
                               'Pachy_MEASURE_KappaAngle':[0,25],            # kappa angle
                               'Pachy_MEASURE_Thickness':[400,600],          # pachimetry
                               'Tono_MEASURE_Average':[],                    # tonometry
                               'Topo_KERATOCONUS_Kpi':[],                    # keratoconus index
                               'AnteriorChamberVol':[],                      # computed
                               'AcdToPupilRadius_Ratio':[],                  # computed
                               'PupilRadiusToW2W_Ratio':[],                  # computed
                               'Topo_GENERAL_Geo_Q':[],
                               'kRatio':[],                                  # computed
                               'J0_3':[],                                      # computed
                               'J45_3':[],                                     # computed
                               'BlurStrength_3':[]}                             # computed

        # self.featureList = np.asanyarray(self.featureList)

    def Load(self,fileName):
        """
            Load a joint VX-EMR database

            Parameters:
            -----------
            fileName: str
                a full path to a csv EMR-VX120 joint database
        """
        if not fileName.__class__==str:
            raise ValueError('fileName must be a string. Got {fileName.__class__}')

        with open(fileName) as csvFile:
            self.data = pd.read_csv(csvFile,low_memory=False)
        self.dataLoaded = True
        self.dataParsed = False

        self.Left  = pd.DataFrame()
        self.Right = pd.DataFrame()
        self.Both  = pd.DataFrame()

    def GetCumulativeDelta(self,target='se',bins=np.arange(0,2,0.25)):
        """
         Compute the cumulative proportion of the delta being lower than d, where d
         an increasing value in jumps of 0.25 diopters

         Parameters:
         ------------
         target : {'se,'sphere','cylinder','glassesSphere','glassesCylinder'}, default='se'
            'se'- for the spherical equivalent delta
            'cylinderDelta' - for the cylinder delta
            'sphereDelta' - for the spherical equivalent delta
            'glassesSphere' - for the current glasses sphere delta
            'glassesCylinder' - for the current glasses cylinder delta

         Output:
         --------
         res - a table of cumulative percentages from the current data and current target below delta, as defined by the bins
               for left right and both eyes
        """
        if np.isin(target.lower(),['se','sphericalequivalent']):
            t = 'SphericalEqDelta'
        elif target.lower() =='cylinder':
            t = 'CylinderDelta'
        elif target.lower()=='sphere':
            t = 'SphereDelta'
        elif target.lower()=='glassessphere':
            t = 'GlassesSphereDelta'
        elif target.lower()=='glassescylinder':
            t = 'GlassesCylinderDelta'
        else:
            raise Exception('option {} is not supported'.format(target))
        res = pd.DataFrame()
        lInds = np.where(pd.isnull(self.Left[t])==False)[0]
        rInds = np.where(pd.isnull(self.Right[t])==False)[0]
        bInds = np.where(pd.isnull(self.Both[t])==False)[0]
        for sIdx in range(len(bins)):
            res.loc[sIdx,target+'_delta'] = bins[sIdx]
            res.loc[sIdx,'Left']  = len(np.where(np.abs(self.Left.loc[self.Left.index[lInds],t])<=bins[sIdx])[0])/len(lInds)
            res.loc[sIdx,'Right'] = len(np.where(np.abs(self.Right.loc[self.Right.index[rInds],t])<=bins[sIdx])[0])/len(rInds)
            res.loc[sIdx,'Both']  = len(np.where(np.abs(self.Both.loc[self.Both.index[bInds],t])<=bins[sIdx])[0])/len(self.Both)
        return res

    def GetTrainingAndValidationIndices(self,labels,trainProp=0.8,equalizeTrainingClasses=False):
        """
            Assign indices randomly into training and testing data
            return nonoverlapping training, testing fully covering numCases: len(training)+len(testing) = numCases
            Parameters:
            -------------
            labels - vector
                a binary label vectors with one label per observation
            trainProp, float, defaul=0.8
                proportion of observations assigned for training out of the total. the reminder is labeled for testing
            equalizeTrainingClasses : bool, default=False
                make equal the proportion of each class in the training set
            Output:
            -------
            training : vector
                indices of training set
            testing : vector
                indices of testing set
        """
        numCases = len(labels)
        assert((trainProp<1.0)&(trainProp>0.0))
        numTraining = np.round(trainProp*numCases).astype(np.int)
        rp          = np.random.permutation(numCases)
        training    = rp[:numTraining]
        testing     = rp[numTraining:]

        if equalizeTrainingClasses:
            # indices for each class
            n0 = training[np.where(labels[training]==0)[0]]
            n1 = training[np.where(labels[training]==1)[0]]
            if len(n0)>=len(n1):
                n0 = n0[:len(n1)] # truncate no to match length of n1
            else:
                n1 = n1[:len(n0)] # truncate n1 to match length of n0
            training = np.append(n0,n1)

        return training, testing

    def Parse(self,handleMissing='impute',imputeMethod='univariate',nNeighbors=2,weights='distance',vertexDistance=0.012):
        """
            Parse a merged (joint) db of the vx120 and its matched EMR entries


            Parameters:
            -----------
            handleMissing: {'discard','impute'}, str, default='impute
                either discard rows with missing data or impute, fill in missing with mean columnd values
            univariate: {True, False}, bool, default=True
                for handleMissing='impute', either  univariate imputation based on column values
                or multivariate, based on the whole feature matrix
        """
        if not self.dataParsed:

            # impute missing values using the median
            self.data  = self.data.fillna(self.data.median(axis=0))
            print('[Classifier] imputing missing values')

            self.data = self.transformer.Transform(self.data)
            print('[Classifier] Transforming data')

            # computing additional features
            for eIdx in ['_Left','_Right']:
                # ACD to pupil radius
                self.data[f'AcdToPupilRadius_Ratio{eIdx}'] = 0.5*self.data[f'Pachy_MEASURE_Acd{eIdx}']\
                                                                /self.data[f'WF_RT_Fv_Meso_PupilRadius{eIdx}']
                # pupil radius to wtw ratio
                self.data[f'PupilRadiusToW2W_Ratio{eIdx}'] = 2*self.data[f'WF_RT_Fv_Meso_PupilRadius{eIdx}']\
                                                            /self.data[f'Pachy_MEASURE_WhiteToWhite{eIdx}']
                self.data[f'kRatio{eIdx}']                 = self.data[f'Topo_Sim_K_K1{eIdx}']/self.data[f'Topo_Sim_K_K2{eIdx}']

            # divide the data to Left and Right according to the featureList
            for fIdx in self.featureList.keys():
                if fIdx+'_Left' in self.data.keys():
                    self.Left.loc[:,fIdx] = self.data[fIdx+'_Left']
                if fIdx+'_Right' in self.data.keys():
                    self.Right.loc[:,fIdx] = self.data[fIdx+'_Right']
                if fIdx in self.data.keys():
                    self.Left.loc[:,fIdx]  = self.data[fIdx]
                    self.Right.loc[:,fIdx] = self.data[fIdx]
            print('[Classifier] dividing data to Left and Right eye')

            # Add subjective values to Left and Right databases
            self.Left['EMR:VisualAcuitySphere']    = self.data['EMR:VisualAcuitySphere_Left'].apply(self.transformer.Round)
            self.Left['EMR:VisualAcuityCylinder']  = self.data['EMR:VisualAcuityCylinder_Left'].apply(self.transformer.Round)
            self.Left['EMR:VisualAcuityAxis']      = self.data['EMR:VisualAcuityAxis_Left'].apply(self.transformer.Round,args=[1])
            self.Right['EMR:VisualAcuitySphere']   = self.data['EMR:VisualAcuitySphere_Right'].apply(self.transformer.Round)
            self.Right['EMR:VisualAcuityCylinder'] = self.data['EMR:VisualAcuityCylinder_Right'].apply(self.transformer.Round)
            self.Right['EMR:VisualAcuityAxis']     = self.data['EMR:VisualAcuityAxis_Right'].apply(self.transformer.Round,args=[1])

            # Compute cylinder delta (objective-subjective)
            self.Left['CylinderDelta']  = (self.Left['WF_CYLINDER_R_3']  -
                                           self.Left['EMR:VisualAcuityCylinder'])
            self.Right['CylinderDelta'] = (self.Right['WF_CYLINDER_R_3'] -
                                           self.Right['EMR:VisualAcuityCylinder'])

            # Compute sphere delta
            self.Left['SphereDelta']  = (self.Left['WF_SPHERE_R_3'] -
                                         self.Left['EMR:VisualAcuitySphere'])
            self.Right['SphereDelta'] = (self.Right['WF_SPHERE_R_3'] -
                                         self.Right['EMR:VisualAcuitySphere'])

            # compute spherical equivalent
            self.Left['ObjectiveSphericalEquivalent']   = (self.Left['WF_SPHERE_R_3']+
                                                           self.Left['WF_CYLINDER_R_3']/2)
            self.Left['SubjectiveSphericalEquivalent']  = (self.Left['EMR:VisualAcuitySphere'] +
                                                           self.Left['EMR:VisualAcuityCylinder']/2)
            self.Right['ObjectiveSphericalEquivalent']  = (self.Right['WF_SPHERE_R_3']+
                                                           self.Right['WF_CYLINDER_R_3']/2)
            self.Right['SubjectiveSphericalEquivalent'] = (self.Right['EMR:VisualAcuitySphere'] +
                                                           self.Right['EMR:VisualAcuityCylinder']/2)

            # Compute spherical equivalent delta
            self.Left['SphericalEqDelta']  =  (self.Left['ObjectiveSphericalEquivalent'] -
                                               self.Left['SubjectiveSphericalEquivalent'])
            self.Right['SphericalEqDelta'] =  (self.Right['ObjectiveSphericalEquivalent'] -
                                               self.Right['SubjectiveSphericalEquivalent'])

            # Filter out features outside the input ranges
            validIndsL = np.ones(shape=len(self.Left),dtype=np.bool)
            validIndsR = np.ones(shape=len(self.Right),dtype=np.bool)
            for dk,dv in self.featureList.items():
                if len(dv)>0:
                    newIndsL   = (self.Left[dk]<=np.max(dv))&(self.Left[dk]>=np.min(dv))
                    newIndsR   = (self.Right[dk]<=np.max(dv))&(self.Right[dk]>=np.min(dv))
                    # print(f'{dk} numValidL: {newIndsL.sum()}')
                    # print(f'{dk} numValidR: {newIndsR.sum()}')
                    validIndsL = validIndsL&newIndsL
                    validIndsR = validIndsR&newIndsR
            self.Left  = self.Left.loc[validIndsL]
            self.Right = self.Right.loc[validIndsR]

            # Rearrange indices =
            self.Left.index  = range(len(self.Left))
            self.Right.index = range(len(self.Right))

            # Append for Both eyes
            self.Both = self.Left.append(self.Right)
            # Rearrange the indices for Both
            self.Both.index = range(len(self.Both))

            self.dataParsed = True

    def GetLabelsFromDeltas(self,data,classes,groupEndBins=True):
        '''
         assign laels to the data in data according to classes in classes
         Parameters:
         ----------
         data
        '''
        labels = data.apply(self._AssignLabel,args=[classes,groupEndBins]).to_frame()
        labels.index = data.index
        return labels

    def GetClasses(self,data,predict='correction',target='se',
                   delta=0.25,sphereDelta=None,cylinderDelta=None,groupEndBins=False,discardUnlabeled=False):
        """
         Get labels for eyes based on the Delta set.

         Parameters:
         -----------
         predict: {'deltaSet','correction'},str
            labels for the prediction of delta-set or correction
         target : {'seDelta', 'cylinderDelta', 'sphereDelta' }, default='seDelta
            the Delta set
         eye : {'right', 'left','both' } , default='both'
            data to use
         delta : float, default=0.25
            a positive sclar defining the Delta set by |Delta|<=delta
         groupEndBins: bool, default=False
            observations lower/higher than min/max(cylinderDelta) or min/max(sphereDelta) are assigned end class labels
         discardUnlabeled: bool, default=False
            if set to True observation with labels not matching any of the vlaued of cylinder or sphereDelta are discarded
            otherwise, unlabeled observations are grouped and assigned the integer label numClasses+1

         Output:
         ---------
         classes :
             for predict='deltaset'
                a binary vector the same size as self.target with 1 for |Delta|<=delta , 0 otherwise
             for predict = 'correction'
               a vector of integer classes indicating the correction from objective to subjective
               the class number correspond to the correction needed in the array of sphDelta or cylDelta
        """

        if self.dataParsed:
            # data = self.GetData(eye=eye)
            if predict.lower()=='deltaset':
                if target.lower()=='se':
                    classes = np.where(np.abs(data['SphericalEqDelta'])<=delta,1,0)
                elif target.lower()=='cylinder':
                    classes = np.where(np.abs(data['CylinderDelta'])<=delta,1,0)
                elif target.lower()=='sphere':
                    classes = np.where(np.abs(data['SphereDelta'])<=delta,1,0)
                else:
                    raise Exception('Option {} is not supported'.format(target))

            elif predict.lower()=='correction':
                classes = np.zeros(len(data),dtype=np.int)
                if target.lower()=='cylinderdelta':
                    numClasses = len(cylinderDelta)
                    if cylinderDelta==None:
                        raise Exception('cylinderDelta bins must be defined')
                    for lIdx in range(len(data)):
                        try:
                            if groupEndBins:
                                if data.loc[data.index[lIdx],'CylinderDelta']<np.min(cylinderDelta):
                                    classes[lIdx] = 0
                                elif data.loc[data.index[lIdx],'CylinderDelta']>np.max(cylinderDelta):
                                    classes[lIdx] = numClasses-1
                                else:
                                    classes[lIdx] = cylinderDelta.index(data.loc[data.index[lIdx],'CylinderDelta'])
                            else:
                                classes[lIdx] = cylinderDelta.index(data.loc[data.index[lIdx],'CylinderDelta'])
                        except:
                    #                            classes[lIdx] = numClasses
                            classes[lIdx] = -1
                elif target.lower()=='cylinder':
                    numClasses =len(cylinderDelta)
                    if cylinderDelta==None:
                        raise Exception('cylinderDelta bins must be defined')
                    for lIdx in range(len(data)):
                        try:
                            classes[lIdx] = cylinderDelta.index(data.loc[lIdx,'EMR:VisualAcuityCylinder'])
                        except:
                            classes[lIdx] = -1

                elif target.lower()=='spheredelta':
                    if sphereDelta==None:
                        raise Exception('sphere delta must be defined')
                    for lIdx in range(len(data)):
                        try:
                            if groupEndBins:
                                if data.loc[data.index[lIdx],'SphereDelta']<np.min(sphereDelta):
                                    classes[lIdx] = 0
                                elif data.loc[data.index[lIdx],'SphereDelta']>np.max(sphereDelta):
                                    classes[lIdx] = numClasses-1
                                else:
                                    classes[lIdx] = sphereDelta.index(data.loc[data.index[lIdx],'SphereDelta'])
                            else:
                                classes[lIdx] = sphereDelta.index(data.loc[data.index[lIdx],'SphereDelta'])
                        except:
                            classes[lIdx] = -1
                elif target.lower()=='sphere':
                    if sphereDelta==None:
                        raise Exception('sphere delta must be defined')
                    for lIdx in range(len(data)):
                        try:
                            classes[lIdx] = sphereDelta.index(data.loc[data.index[lIdx],'EMR:VisualAcuitySphere'])
                        except:
                            classes[lIdx] = -1
                elif target.lower()=='both':
                    if sphereDelta==None or cylinderDelta==None:
                        raise Exception('sphreDelta and cylinderDelta must be defined')
                    correctionPairs = list(itertools.product(sphereDelta,cylinderDelta))
                    # Assign labels based on the correction pairs
                    for pIdx in range(len(data.index)):
                        try:
                            classes[pIdx] = correctionPairs.index(data.loc[data.index[pIdx],'SphereDelta'],data.loc[data.index[pIdx],'CylinderDelta'])
                        except:
                            classes[pIdx] = -1
                else:
                    raise Exception('Option target={} is not supported'.format(target))
            return classes
        else:
            print('Data not loaded or not parsed')
            return None

    @ staticmethod
    def _AssignLabel(val,classes,groupEndBins=True):
        '''
         Assign labels from classes to a values
         val-, float
          the delta between objective and subjective (spher or cylinder)
         classes, list
           a list of possible deltas, the order in the list will determine the class number
         groupEndBins, bool, default=True
          if val<min(classes) it will be assigned 0
          if val>max(classes) it will be assigned len(classes)-1
         Output:
         -------
         label, float
           a number corresponding to the class in classes

        '''

        if groupEndBins:
            m = min(classes)
            M = max(classes)
            if val<m:
                return 0
            elif val>M:
                return len(classes)-1
            else:
                try:
                    return classes.index(val)
                except:
                    return -1
        else:
            try:
                return classes.index(val)
            except:
                return -1

    def GetFeatureMatrix(self,data, features='all',random=False):
        """
            Get feature matrix.

            Parameters:
            ------------
            features : {'all',list}, default='auto'
                'all' - use all features in self.featureList
                 or a list of indices from self.featureList
            eye : {'left', 'right', 'both'}, default='both'
                data to use
            autoSelectFeatures: {True,False}, bool, default=False
                automatic feature selection
            Output:
            ---------
            fMat- dataFrame
                feature matrix with columns corresponding to features and rows for observations
        """
        featureMat = data
        featureMat = featureMat.loc[:,self.featureList.keys()]
        if random:
            featureMat = self.GenerateRandomPatient(N=len(featureMat))

        # Feature indices to use
        if features.__class__ is str:
            if features.lower()=='all':
                fInds =  range(len(self.featureList.keys()))
            else:
                raise Exception('option features={} is not supported'.format(features))
        elif features.__class__ is list:
            fInds = features
        else:
            raise Exception('option features={} is not supported'.format(features))

        # Set features used
        self.featuresUsed = np.asanyarray(list(self.featureList.keys()))[fInds]

        # Return a feature matrix
        return featureMat.loc[:,self.featuresUsed]

    def PredictDeltaSet(self,classifier='rf',target='se',
                        eye='both',numTrials=10,delta=0.25,trainProp=0.8,
                        features='all',autoSelectFeatures = False,
                        equalizeTrainingClasses=False,
                        classifierParams=dict(),alpha=0.05):
        """
            Binary classification to predict inclusion in the Delta set
            based on selected prognostic factors. The Delta set
            is either the spherical equivalent delta, cylinder or sphere delta.
            In all cases, patient within Delta set are those for which |Delta|<=delta,
            with delta a positive scalar.

            Parameters:
            -----------
            classifier: {'rf','ab','dt','lr','svm','lda','qda','nb','nn'}, str, default=knn
                rf- random forest,
                ab-adaboost,
                dt- decision tree,
                lr- logistic regression,
                svm- support vector machine,
                lda- linear discriminant analysis,
                qda- quadratic discriminant analysis
                nb- naive bayes,
                nn- neural network
                knn - k-nearest neighbors
            target: {'se','sphere','cylinder'},str, default='seDelta'
                Defining the Delta set
                'se'      - spherical equivalent delta;
                'sphere'  - sphere delta, or sphere correction
                'cylinder'- cylinder delta, or cylinder correction
                the delta is defined as the objective minus subjective
            eye: {'left','right','both'}, str, default='both'
                data to use for classification
            numTrials: int, default=10
                positive integer defining the number of rounds to run classifier
            delta: float, default = 0.25
                positive scalar defining the set |Delta|<=delta
            trainProp: float, default=0.8
                training proportion, positive scalar smaller than 1
            features: {'all','auto', list}, default='auto'
                'all'- use all features in self.featureList
                'auto'- selects features automatically by KS statistics ath the confidence level of 1-alpha
            standardize: bool, default=True
                if True, subtract the mean and divide each feature by its STD
            equalizeTrainingClasses: bool, default=False
                make equal the proportion of each class during training
            alpha: float, default=0.05
                the KS confidence = 1-alpha, in case autoSelectFeatures=True
        """
        # Get labels
        data     = self.GetData(eye=eye)
        patClass = self.GetClasses(data,predict='deltaSet',delta=delta,target=target)
        fMat     = self.GetFeatureMatrix(data,features=features,random=False)
        if autoSelectFeatures:
            # select features (columns of the feature matrix) by KS test
            fMat = self.SelectFeatures(fMat,patClass,alpha=alpha)

        # Preallocations
        results     = pd.DataFrame()
        success     = np.zeros(numTrials)
        failed      = np.zeros(numTrials)
        sucPer      = np.zeros(numTrials)
        score       = np.zeros(numTrials)
        featureRank = np.zeros(shape=fMat.shape[1])
        meanCoeffs  = np.ndarray(shape=fMat.shape[1])

        for tIdx in range(numTrials):
            print("Testing round {}/{}".format(tIdx+1,numTrials))

            # Construct a classifier for the current test
            model = self.GetClassifier(params=classifierParams)
            # Randomly divide the data into training and testing sub-sets
            trainIdx,testIdx  = self.GetTrainingAndValidationIndices(patClass,trainProp=trainProp,
                                                                     equalizeTrainingClasses=equalizeTrainingClasses)
            featureMatTrain   = fMat.loc[fMat.index[trainIdx],self.featuresUsed].values
            featureMatTest    = fMat.loc[fMat.index[testIdx],self.featuresUsed].values
            if fMat.shape[1]==1:
                # Reshape to meet function requirements
                featureMatTrain = featureMatTrain.reshape(-1,1)
                featureMatTest  = featureMatTest.reshape(-1,1)
            labelMatTrain = patClass[trainIdx]
            labelMatTest  = patClass[testIdx]

            # Fit model on training data
            model.fit(featureMatTrain, labelMatTrain)

            # Predict
            predictedLabels = model.predict(featureMatTest)
            score[tIdx]     = model.score(featureMatTest,labelMatTest)
            # Collect results
            success[tIdx] = sum(predictedLabels==labelMatTest)
            failed[tIdx]  = len(labelMatTest)-success[tIdx]
            sucPer[tIdx]  = 100*success[tIdx]/len(testIdx)

        # Result summary to return
        results.loc[0,'succ_mean'] = np.mean(sucPer)
        results.loc[0,'succ_std']  = np.std(sucPer)
        results.loc[0,'succ_max']  = np.max(sucPer)
        results.loc[0,'succ_min']  = np.min(sucPer)
        results.loc[0,'mean_score']    = np.mean(score)
        results.loc[0,'N_train']       = len(trainIdx)
        results.loc[0,'N_test']        = len(testIdx)
        results.loc[0,'N_in_train']    = sum(labelMatTrain)
        results.loc[0,'N_in_test']     = sum(labelMatTest)
        results.loc[0,'N_succ_mean']   = np.mean(success)
        results.loc[0,'N_succ_std']    = np.std(success)
        results.loc[0,'N_fail_mean']   = np.mean(failed)
        results.loc[0,'N_fail_std']    = np.std(failed)

        # Construct a histogram of importance of features
        if np.isin(classifier.lower(),['logisticregression','lr']):
            self.featureRank = featureRank/(np.sum(featureRank)*numTrials) # normalize
            self.meanCoeffs  = meanCoeffs/numTrials             # mean values
        elif np.isin(classifier.lower(),['decisiontree','dt','randomforest','rf']):
            self.featureRank  = model.feature_importances_

        return results

    def GenerateRandomPatient(self,N=1,eye='both'):
        """
            Generate a random patient by sampling feature values from empirical feature distribution
        """
        fMat = pd.DataFrame()
        data = self.GetData(eye=eye)
        for f in self.featureList.keys():
            # compute feature empirical distribution
            # uVals = list(np.unique(data[f]))
            p     = np.zeros(len(data[f]))
            for uIdx in range(len(data[f])):
                p[uIdx] = np.sum(data[f]==data.loc[uIdx,f])
            p = p/np.sum(p)

            # randomly select values for feature based on the empirical distribution
            fMat[f] = np.random.choice(data[f],size=N,p=p)
        return fMat

    def PredictCorrection(self,eye='both',target='cylinder',
                          numTrials=5,classifier='rf',
                          trainProp=0.8,
                          sphereDelta=[-0.5,-0.25, 0.0, 0.25],
                          cylinderDelta=[-0.25, 0.0],
                          equalizeTrainingClasses=False,
                          classifierParams={},features='all',
                          autoSelectFeatures=False,alpha=0.05,
                          featureSelectionMode='fpr',
                          rejProp=0.8,
                          groupEndBins=False):
        """
            Predict the sphere-cylinder correction pairs.

            Parameters:
            -----------
            eye: {'left,'right','both'},str default='both'
                data to use
            target: {'sphere','cylinder','both'}, str, default='cylinder'
                classify into cylinder correction classes, sphere correction classes,
                or all combinations of sphere and cylinder pairs defained by sphereDelta, cylinderDelta
                for cylinderDelta, the sphereDelta range is ignored;
                for sphereDelta, the cylinderDelta range is ignored.
            numTrials: int, default=5
                number of times to run the classifier
            classifier: {'lr','dt','knn','nb','nn','lda','qda','rf'}, str, default='rf'
                classifier to use. see Getclassifier for options
            standardize: {True,False}, bool, default=True
                standardize the trining and validation data
            trainProp: float, default=0.9
                proportion of the data used for training
            equalizeTrainingClasses: bool, default=False
                make the size of classes equal in the training data
            autoSelectFeatures: bool. default=True
                automatically select features by pairwise kolmogorov smirnoff test
            alpha: float, default=0.05
                the confidence level of the KS test conf=1-alpha, for which to reject the null hypothesis
                is case autoSelectFeatures=True
            rejProp: float, default==0.8
                in pairswise KS test, select those features which their null hypothesis is
                rejected at least round(rejProp*(numClasses-1)) times
            groupEndBins: bool, default=True
                observations with delta<min or >max are assigned end bins' labels
            Output:
            --------
            res: dataframe
                statistics and accuracy results for the classification
            confMat: array
                confusion matrix
        """

        data   = self.GetData(eye=eye) # Get parsed data

        # This gives a total of |cylinderDelta|x|sphereDelta| number of classes
        # Discard all patients for whom the correction (delta) is not in the correction pairs
        fMat   = self.GetFeatureMatrix(data,features=features)

        labels = self.GetClasses(data,predict='correction',
                                      target=target,
                                      sphereDelta=sphereDelta,
                                      cylinderDelta=cylinderDelta,
                                      groupEndBins=groupEndBins)

        # Discard unlabeled, truncate the labels and the feature matrix accordingly
        print(f'Discarding {sum(labels==-1)} unlabeled observations ({100*sum(labels==-1)/len(labels)}%, remaining {sum(labels!=-1)})')
        validInds       = np.where(labels!=-1)[0]
        labels          = labels[validInds]
        fMat            = fMat.iloc[validInds]
        data            = data.iloc[validInds]
        numClasses      = len(np.unique(labels))
        correctionPairs = list(itertools.product(sphereDelta,cylinderDelta))

        # Select features automatically by a pairwise ks test
        if autoSelectFeatures:
            fMat = self.SelectFeatures(fMat,labels,alpha=alpha,mode=featureSelectionMode)
        # Construct empirical distributions of classes to draw samples from
        classProb = np.zeros(numClasses,dtype=np.int)
        for pIdx in range(numClasses):
            classProb[pIdx] = np.sum(labels==pIdx)
        classProb = classProb/sum(classProb)

        # Previous objective and subjective sphere and cylinder
        objSphere   = data['WF_SPHERE_R_3']
        subSphere   = data['EMR:VisualAcuitySphere']
        objCylinder = data['WF_CYLINDER_R_3']
        subCylinder = data['EMR:VisualAcuityCylinder']
        sphDelta    = objSphere   - subSphere
        cylDelta    = objCylinder - subCylinder

        # Preallocations
        res              = pd.DataFrame()
        accuracy         = np.zeros(numTrials, dtype=np.float)
        accuracyRand     = np.zeros(numTrials, dtype=np.float)
        delta0_25_before = np.zeros(numTrials, dtype=np.float)
        delta0_5_before  = np.zeros(numTrials, dtype=np.float)
        delta0_25_after  = np.zeros(numTrials, dtype=np.float)
        delta0_5_after   = np.zeros(numTrials, dtype=np.float)
        deltaRand0_25    = np.zeros(numTrials, dtype=np.float)
        deltaRand0_5     = np.zeros(numTrials, dtype=np.float)
        confMat          = np.zeros(shape=(numClasses,numClasses),dtype=np.float)
        # featureRank      = np.zeros(shape=fMat.shape[1])
        # Main loop - classifiy
        for tIdx in range(numTrials):
            # Construct a model to predict the delta sphere
            print(f'Trial {tIdx+1}/{numTrials}')
            trainIdx, testIdx = self.GetTrainingAndValidationIndices(labels,trainProp=trainProp,
                                                                    equalizeTrainingClasses=equalizeTrainingClasses)
            # Get feature matrix and labels for the current trial
            featureMatTrain = fMat.loc[fMat.index[trainIdx]]
            featureMatTest  = fMat.loc[fMat.index[testIdx]]
            labelsTrain     = labels[trainIdx]
            labelsTest      = labels[testIdx]

            # Get classifier for the current trial
            model = self.GetClassifier(params=classifierParams)
            # Train
            model.fit(featureMatTrain,labelsTrain)

            # Predict correction
            predictedLabels = model.predict(featureMatTest)
            oldSph          = objSphere.loc[objSphere.index[testIdx]].values
            oldCyl          = objCylinder.loc[objCylinder.index[testIdx]].values

            seDeltaOld      = sphDelta.loc[sphDelta.index[testIdx]]+cylDelta.loc[cylDelta.index[testIdx]]/2
            newSph          = np.zeros(len(testIdx))
            newCyl          = np.zeros(len(testIdx))

            accuracy[tIdx] = 100*model.score(featureMatTest,labelsTest)
            # Compute the confusion matrix for this trial
            for prlIdx in range(len(predictedLabels)):
                confMat[labelsTest[prlIdx],predictedLabels[prlIdx]]+=1
                if target.lower()=='cylinderdelta':
                    newCyl[prlIdx] = oldCyl[prlIdx] - cylinderDelta[predictedLabels[prlIdx]]
                elif target.lower()=='cylinder':
                    newCyl[prlIdx] = cylinderDelta[predictedLabels[prlIdx]]
                elif target.lower()=='spheredelta':
                    newSph[prlIdx] = oldSph[prlIdx] - sphereDelta[predictedLabels[prlIdx]]
                elif target.lower()=='sphere':
                    newSph[prlIdx] = sphereDelta[predictedLabels[prlIdx]]
                elif target.lower()=='both':
                    for prlIdx in range(len(testIdx)):
                        newSph[prlIdx]= oldSph[prlIdx] - correctionPairs[predictedLabels[prlIdx]][0]
                        newCyl[prlIdx]= oldCyl[prlIdx] - correctionPairs[predictedLabels[prlIdx]][1]
                else:
                    raise Exception(f'option: {target} is not supported')

            # Assign the correction to the objective and compute the spherical Eq delta
            if np.isin(target.lower(),['cylinder','cylinderdelta']):
                deltaNew               = newCyl - subCylinder.loc[subCylinder.index[testIdx]].values
                deltaOld               = cylDelta.loc[cylDelta.index[testIdx]]
                delta0_25_after[tIdx]  = len(np.where(np.abs(deltaNew)<=0.25)[0])/len(deltaNew)
                delta0_5_after[tIdx]   = len(np.where(np.abs(deltaNew)<=0.5)[0])/len(deltaNew)
                delta0_25_before[tIdx] = len(np.where(np.abs(deltaOld)<=0.25)[0])/len(deltaOld)
                delta0_5_before[tIdx]  = len(np.where(np.abs(deltaOld)<=0.5)[0])/len(deltaOld)
            elif np.isin(target.lower(),['sphere','spheredelta']):
                deltaNew               = newSph - subSphere.loc[subSphere.index[testIdx]].values
                deltaOld               = sphDelta.loc[sphDelta.index[testIdx]]
                delta0_25_after[tIdx]  = len(np.where(np.abs(deltaNew)<=0.25)[0])/len(testIdx)
                delta0_5_after[tIdx]   = len(np.where(np.abs(deltaNew)<=0.5)[0])/len(testIdx)
                delta0_25_before[tIdx] = len(np.where(np.abs(deltaOld)<=0.25)[0])/len(testIdx)
                delta0_5_before[tIdx]  = len(np.where(np.abs(deltaOld)<=0.5)[0])/len(testIdx)
            elif target.lower()=='both':
                deltaNew               = newSph+newCyl/2 - (oldSph+oldCyl/2)
                deltaOld               = seDeltaOld
                delta0_25_after[tIdx]  = len(np.where(np.abs(deltaNew)<=0.25)[0])/len(testIdx)
                delta0_5_after[tIdx]   = len(np.where(np.abs(deltaNew)<=0.5)[0])/len(testIdx)
                delta0_25_before[tIdx] = len(np.where(np.abs(deltaOld)<=0.25)[0])/len(testIdx)
                delta0_5_before[tIdx]  = len(np.where(np.abs(deltaOld)<=0.5)[0])/len(testIdx)


            # sphEqDeltaNew         = newSph+newCyl/2
            # delta0_25_after[tIdx] = len(np.where(np.abs(sphEqDeltaNew)<=0.25)[0])/len(testIdx)
            # delta0_5_after[tIdx]  = len(np.where(np.abs(sphEqDeltaNew)<=0.5)[0])/len(testIdx)

            # draw correction pairs by random samples for the empirical distribution
            randSphCorr = np.zeros(len(testIdx),dtype=np.float)
            randCylCorr = np.zeros(len(testIdx),dtype=np.float)
            rndSucc     = np.zeros(len(testIdx),dtype=np.int)
            for pIdx in range(len(testIdx)):
                corrInd           = np.random.choice(range(numClasses),p=classProb)
                rndSucc[pIdx]     = corrInd==labelsTest[pIdx]
                if target.lower()=='cylinder':
                    randCylCorr[pIdx] = cylinderDelta[corrInd]
                    randSphCorr[pIdx] = 0
                elif target.lower()=='sphere':
                    randSphCorr[pIdx] = sphereDelta[corrInd]
                    randCylCorr[pIdx] = 0
                elif target.lower()=='both':
                    randSphCorr[pIdx] = correctionPairs[corrInd][0]
                    randCylCorr[pIdx] = correctionPairs[corrInd][1]

            newSphRand          = oldSph - randSphCorr
            newCylRand          = oldCyl - randCylCorr
            accuracyRand[tIdx]  = 100*sum(rndSucc)/len(testIdx)
            sphEqRand           = (newSphRand) + (newCylRand)/2
            deltaRand0_25[tIdx] = len(np.where(sphEqRand<=0.25)[0])/len(testIdx)
            deltaRand0_5[tIdx]  = len(np.where(sphEqRand<=0.5)[0])/len(testIdx)

        # Construct a histogram of importance of features
        self.featureRank  = model.feature_importances_

        # Summarize results
        res.loc[0,'succ_mean']              = np.mean(accuracy)
        res.loc[0,'succ_min']               = np.min(accuracy)
        res.loc[0,'succ_max']               = np.max(accuracy)
        res.loc[0,'mean_delta_0_25_before'] = np.mean(delta0_25_before)
        res.loc[0,'mean_delta_0_25_after']  = np.mean(delta0_25_after)
        res.loc[0,'mean_delta_0_5_before']  = np.mean(delta0_5_before)
        res.loc[0,'mean_delta_0_5_after']   = np.mean(delta0_5_after)
        res.loc[0,'mean_delta_0_25_rand']   = np.mean(deltaRand0_25)
        res.loc[0,'mean_delta_0_5_rand']    = np.mean(deltaRand0_5)
        res.loc[0,'succ_rand']              = np.mean(accuracyRand)
        res.loc[0,'succ_std']               = np.std(accuracy)
        res.loc[0,'N']                      = len(labels)
        res.loc[0,'N_train']                = len(trainIdx)
        res.loc[0,'N_test']                 = len(testIdx)

        #        for cIdx in range(len(confMat)):
        #            confMat[cIdx] = confMat[cIdx]/np.sum(confMat)
        return res, confMat/np.sum(confMat)

    def GetData(self,eye='both'):
        """
             Get data from left right or both eyes
             Parameters:
             --------
                eye: {'left','right','both'}

        """
        if eye.lower()=='left':
            data = self.Left.copy()
        elif eye.lower()=='right':
            data = self.Right.copy()
        elif eye.lower()=='both':
            data = self.Both.copy()
        else:
            raise Exception('option eye={} is nnot supported'.format(eye))

        return data

    def TuneHyperParameters(self,classifier='rf',paramDomain=None,
                            target='cylinder',predict='correction',
                            cylinderDelta = [-1,-0.75,-0.5,-0.25,0.0],
                            sphereDelta   = [-0.75,-0.5,-0.25,0,0.25,0.5,0.75],
                            eye='both', features='all',
                            searchType = 'random',
                            n_iter = 100,crossValidations=3):
        """
         Use the hyperparameter tuning method in sklearn to tune parameters
         and obtained a trained classifier
         Parameters:
         ------------
         paramDomain: dictionary, default=None
            a dictionary with names of parameters as keys and values to check as values
            A sample parameter domain can be obtained from self.GenerateParameterDomain
         target: {}

        """
        if paramDomain is None:
            paramDomain = self.GenerateParameterDomain(classifier=classifier) # use default parameter domain

        model = self.GetClassifier()
        # prepare data
        if eye.lower()=='left':
            data = self.Left
        elif eye.lower()=='right':
            data = self.Right
        elif eye.lower()=='both':
            data = self.Both
        featMat = self.GetFeatureMatrix(data,features=features)
        # get classes
        classes = self.GetClasses(self.Both,predict=predict,target=target,
                                  cylinderDelta=cylinderDelta,
                                  sphereDelta=sphereDelta)

        # truncate missing classes from data
        inds = np.where(classes!=-1)[0]
        print(f"Using {len(inds)}/{len(classes)} ({len(inds)/len(classes)}%) observations")
        data    = data.iloc[inds]
        classes = classes[inds]
        featMat = featMat.iloc[inds]
        # get training and validation indices
        training, testing = self.GetTrainingAndValidationIndices(classes)

        # Random search of parameters, using n-fold cross validation,
        # search across different combinations, and use all available cores
        if searchType.lower()=='random':
            tunedClassifier = RandomizedSearchCV(estimator = model, param_distributions = paramDomain,
                                                n_iter = n_iter, cv = crossValidations,
                                                verbose=2, random_state=42,
                                                n_jobs = -1)# Fit the random search model
        elif searchType.lower()=='grid':
            tunedClassifier = GridSearchCV(estimator = model, param_grid = paramDomain,
                                                cv = crossValidations,
                                                verbose=2,
                                                n_jobs = -1)# Fit the random search model
        tunedClassifier.fit(featMat.iloc[training], classes[training])

        # print statistics
        predictedLabels = tunedClassifier.best_estimator_.predict(featMat.iloc[testing])
        print(f'Classifier Score {tunedClassifier.best_estimator_.score(featMat.iloc[testing],classes[testing])}')
        # Construct confusion matrix
        if target.lower() == 'cylinder':
            confMat = np.zeros(shape=(len(cylinderDelta),len(cylinderDelta)))
        elif target.lower() =='sphere':
            confMat = np.zeros(shape=(len(sphereDelta),len(sphereDelta)))

        groundTruth = classes[testing]
        for pIdx in range(len(predictedLabels)):
            confMat[int(groundTruth[pIdx]),int(predictedLabels[pIdx])]+=1

        print('____ Results____')
        print(f'Accuracy {target}=0: {np.sum(np.diag(confMat))/np.sum(confMat)}')
        print(f'Accuracy {target}<=0.25: {(np.sum(np.diag(confMat))+np.sum(np.diag(confMat,1))+np.sum(np.diag(confMat,-1)))/np.sum(confMat)}')
        print(f'Accuracy {target}<=0.5: {(np.sum(np.diag(confMat))+np.sum(np.diag(confMat,1))+np.sum(np.diag(confMat,-1))+np.sum(np.diag(confMat,2))+np.sum(np.diag(confMat,-2)))/np.sum(confMat)}')

        # Load the classifier class with class names sphere and cylinder delta and feature names
        classifier = tunedClassifier.best_estimator_
        classifier.featuresUsed   = self.featuresUsed
        classifier.sphereDelta    = sphereDelta
        classifier.cylinderDelta  = cylinderDelta
        classifier.predictionType = predict
        return classifier, confMat

    def GridFitClassifier(self,classifier='rf',
                          maximize='min',
                          predict='correction',target='cylinder',
                          delta=0.25,eye='both',
                          numTrials=10,features='all',
                          autoSelectFeatures=False,
                          standardize=False,alpha=0.05,
                          rejProp=0.667,trainProp=0.8,
                          sphereDelta=[-0.5,-0.25,0.0,0.25,0.5],
                          cylinderDelta =[-0.75,-0.5,-0.25,0.0],
                          equalizeTrainingClasses=False):

        """
         DEPRECATED, now replaced by TuneHyperParameters
         Tune hyper-paramters of a classifier training and validating it on each point of a parameter space.

         Parameters:
         --------------
         classifier: list, default=['lr','knn']
            classifiers to
         predict: {'correction','deltaset'}, str, default='correction'
            predict delta set spherical equivalent, cylinder delta or sphere delta,
            as specified by target, or correction, spher correction, cylinder correction
            or both as specified by target.
         target: {'se','cylinder','sphere'}, str, default='cylinder'
            for predict='deltaSet', the spherical equivalent (se), the cylinderDelta (cylinder)
            or the sphereDelta (sphere) are the zero set in the binary classification problem.
            when predict='correction', target determines whether classification is
            performed on cylinder correction using the classes defined in cylinderDelta

         sphereDelta: {list}, default=[-0.5,-0.25,0.0,0.25]
            value of sphere correction to classify to.
            only taken into acount if predict='correction', and target='sphere or 'both'
         cylinderDelta: {list}, default=[-0.25,0.0],
            value of cylinder correction to classify to .
            Only taken into account if predict='correction', and target='cylinder' or 'both'
         delta: float, default=0.25
            defines the Delta set for which |Delta|<=delta
            for predict='deltaSet'
         eye: {'left','right','both'}, str, default='both'
            Perform classification on the data for a specific eye
         maximize: {'min','max','mean}, str, default='min'
            chose those set of parameters which either maximize the
            minimal accuracy (min), average accuracy (mean), or maximal accuracy (max)
         Output:
         --------
         meanParams: dict
            classifier parameter yielding best mean score
         meanScore: float
            classification score obtained using meanParams
         maxParams: dict
            a dictionary with parametes yielding best max score
         maxScore: float
            classification score obtained using maxParams
        """
        if classifier.__class__ is str:
           classifier=[classifier]
        elif classifier.__class__ is not list:
            raise Exception('classifier must be either the name of the classifier or a list of names')

        bestResults = pd.DataFrame() # records from all classifiers
        bestParams  = np.ndarray(shape=len(classifier),dtype=dict)
        cIdx        = 0
        for classIdx in classifier:
            # get classifier specific parameters
            params = self.GetClassifier().get_params()
            domain = self.GenerateParameterDomain(classIdx)
            # Preallocate grid score results
            vals       = list(domain.values())
            attr       = list(domain.keys())
            bestScore  = 0
            print('Testing classifier: {}'.format(classIdx))
            for vInds in list(itertools.product(*vals)):
                # Assign parameters
                for vIdx in range(len(vInds)):
                    params.__setitem__(attr[vIdx],vInds[vIdx])
                    print(f'setting {attr[vIdx]} to {vInds[vIdx]}')
                # Run classifier
                if predict.lower() =='deltaset':
                    results = self.PredictDeltaSet(classifier=classIdx,target=target,
                                        eye=eye,numTrials=numTrials,
                                        delta=delta,features=features,
                                        autoSelectFeatures=autoSelectFeatures,
                                        equalizeTrainingClasses=equalizeTrainingClasses,
                                        # standardize=standardize,
                                        trainProp=trainProp,
                                        classifierParams=params,alpha=alpha)
                elif predict.lower() =='correction':
                    results, _ = self.PredictCorrection(classifier=classIdx,target=target,
                                        numTrials=numTrials,
                                        # standardize=standardize,
                                        autoSelectFeatures=autoSelectFeatures,
                                        trainProp=trainProp,
                                        rejProp=rejProp,
                                        cylinderDelta=cylinderDelta,
                                        sphereDelta=sphereDelta,
                                        equalizeTrainingClasses=equalizeTrainingClasses,
                                        classifierParams=params,alpha=alpha)

                if maximize.lower()=='mean' and results.loc[0,'succ_mean']>bestScore:
                    bestScore   = results.loc[0,'succ_mean']
                    newBestScore = True
                elif maximize.lower()=='min' and results.loc[0,'succ_min']>bestScore:
                    bestScore   = results.loc[0,'succ_min']
                    newBestScore = True
                elif maximize.lower()=='max' and results.loc[0,'succ_max']>bestScore:
                    print('here')
                    bestScore = results.loc[0,'succ_max']
                    newBestScore = True
                else:
                    newBestScore = False

                if newBestScore:
                    bestResults.loc[cIdx,'classifier'] = classIdx
                    bestResults.loc[cIdx,'min']        = results.loc[0,'succ_min']
                    bestResults.loc[cIdx,'mean']       = results.loc[0,'succ_mean']
                    bestResults.loc[cIdx,'max']        = results.loc[0,'succ_max']
                    bestResults.loc[cIdx,'rand']       = results.loc[0,'succ_rand']
                    bestResults.loc[cIdx,'std']        = results.loc[0,'succ_std']
                    # copy classifier parameters
                    bestParams[cIdx]                   = params.copy()
            cIdx+=1
        return bestResults, bestParams

    def GenerateParameterDomain(self, classifier):
        """
            This is a service function to generate a domain of hypoerparameters
            be used in GridFitClassifier for tuning
            Parameters:
            -----------
            classifier: str, classifier name or initials
                classifier name, see GetClassifier for classifier names and initials
            Output:
            ---------
            domain: dict,
                dictioinary with classifier specific parameter names
                and range of values to be tested

        """
        if np.isin(classifier.lower(),['lr','logisticregression']):
            domain = {'C':[0.01, 0.5, 1.0,5.0,10.0],'solver':['newton-cg'],
                    'penalty':['l2','none']}
        elif classifier.lower()=='svm':
            domain = {'C':[0.01,0.1,1.0,5.0,10.0,50.0],
                      'kernel':['poly','rbf','sigmoid'],
                      'degree':[1,2,3],
                      'gamma':['scale']}
        elif np.isin(classifier.lower(),['dt','decisiontree']):
            domain = {'criterion':['gini','entropy'],
                        'max_depth':[10,100,200,300],
                        'max_leaf_nodes':[2,5,10,20],
                        'min_samples_leaf':[10,20,50],
                        'min_samples_split':[2,4,6,10]}
        elif np.isin(classifier.lower(),['rf','randomforest']):
            domain = {'max_depth':[100,200,300,400,500],
                      'n_estimators':[100,200,300,400,500,600,1000],
                      'criterion':['gini','entropy'],
                      'min_samples_split':[2,5,10,20]}
                    #   'min_samples_leaf':[1,5,10]}
        elif np.isin(classifier.lower(),['nn','neuralnetwork']):
            domain = {'activation':['relu','logistic'],
                       'hidden_layer_sizes':[(10,),(10,10,),(10,10,10,),(5,10,20)],
                       'solver':['lbfgs','adam'],
                       'alpha':[1e-1,1e-3,1e-4],
                       'early_stopping':[True,False],
                       'learning_rate':['constant','adaptive']}
        elif np.isin(classifier.lower(),['knn']):
            domain = {'weights':['distance','uniform'],
                    'n_neighbors':[5,10,30,50,80,100],
                    'p':[1,2,3,4],
                    'leaf_size':[2,5,10,20,50],
                    'algorithm':['kd_tree','ball_tree','brute']}
        elif np.isin(classifier.lower(),['nb','naivebayes']):
            domain = {'var_smoothing':[1e-1,1e-2,1e-3,1e-5,1e-9]}
        elif np.isin(classifier.lower(),['ab','adaboost']):
            domain = {'n_estimators':[5,10,20,50,100,150],
                      'learning_rate':[1.0,0.5, 0.1]}
        elif np.isin(classifier.lower(),['qda','quadraticdiscriminantanalysis']):
            domain = {'reg_param':[0.1,1.0,2.0,5.0],
            'store_covariance':[True,False],
            'tol':[1e-1,1e-2,1e-4,1e-5]}
        elif np.isin(classifier.lower(),['lda','lineardiscriminantanalysis']):
            domain = {'solver':['lsqr','eigen'],
                      'shrinkage':[0.01,0.1, 0.5,'auto']}
        else:
            raise Exception('option classifier={} is not supported'.format(classifier))

        return domain

    def GetClassifier(self,params=dict()):
        """
          Construct a classifier save on the class
          Parameters:
          -----------
          classifier: {'lr','svm','dt','rf','nn','nb','ab','qda','lda','knn','bag'}, default:
          knn'
            classifier name or initials:
            logistic regression: 'lr', or 'logisticRegression'
            support vector machine: 'svm'
            Decision tree: 'dt', or 'decisionTree'
            Random forest: 'rf', or 'randomForest'
            Neural network: 'nn', or 'neuralNetwork'
            Naive bayes: 'nb', or 'naiveBayes'
            adaboost': 'ab', or 'adaboost'
            linear discriminant analysis: 'lda', or lineardiscriminantanalysis
            quadratic descriminant analysis: 'qda', or 'quadraticdiscriminantanalysis'
            k nearest-neighbors: 'knn'

          params: dict
             A dictionary with keys as classifier specific parameter and values

          Output:
          -----------
          model -
           A classifier class
        """

        model = RandomForestClassifier(max_depth=200,
                                        n_estimators=500,
                                        min_impurity_split = None,
                                        criterion = 'gini',
                                        # min_samples_split = 45,
                                        # min_samples_leaf= 50,
                                        # max_leaf_nodes = 9,
                                        # bootstrap=True,
                                        # max_features=15,
                                        warm_start=True, # n_estimators increases with each trial to fit new trees
                                        oob_score=True)


        # Set classifier parameters from input params
        if len(params.keys())>0:
            modelParams= list(model.get_params().keys())
            for kIdx in params.keys():
                if np.isin(kIdx,modelParams):
                    model.__setattr__(kIdx,params.get(kIdx))
                else:
                    raise Exception(f'Parameter {kIdx} is not supported for classifier Random forest')
        return model

    def LoadCylinderClassifier(self,modelPath):
        # load a cylinder corection  model
        self.cylModel = pickle.load(open(modelPath, 'rb'))
        self.cylinderModelLoaded = True

    def LoadSphereClassifier(self,modelPath):
        self.sphModel = pickle.load(open(modelPath, 'rb'))
        self.sphereModelLoaded = True

    def PredictSphere(self,features):
        sphClass = None
        if self.sphereModelLoaded:
            sphClass = self.sphModel.predict(np.asanyarray(features).reshape(1,-1))
        return sphClass

    def PredictCylinder(self,features):
        cylClass = None
        if self.cylinderModelLoaded:
            cylClass = self.cylModel.predict(np.asanyarray(features).reshape(1,-1))
        return cylClass

    def ExportClassifier(self,classifier, fileName='trainedClassifier.sav'):
        # export a trained classifier
        # TODO: makek sure the classifier entered is of the right class
        pickle.dump(classifier, open(fileName,'wb'))

    def PlotFeatures(self,target='seDelta',eye='both',delta=0.25,plotType='scatter'):
        """
            Plots features vs. target class either scatter plot or histograms.

            Parameters:
            -----------
            plotType:{'scatter','hist'}, default: 'scatter'
                Type of plot, either scatter of histogram
            eye: {'left', 'right', 'both'}, default: 'both'
                data to use for the plot
            target: {'seDelta', 'sphereDelta', 'cylinderDelta'}, default: 'seDelta'
                the Delta set
            delta {float}, default: 0.25
                positive scalar defining the set |Delta|<= delta.
        """
        data = self.GetData(eye=eye)

        if target.lower()=='sedelta':
            y = data.loc[:,'SphericalEqDelta']
        elif target.lower() =='cylinderdelta':
            y = data.loc[:,'CylinderDelta']
        elif target.lower()=='spheredetla':
            y = data.loc[:,'SphereDelta']
        else:
            raise Exception('option target={} is not supported'.format(target))

        indsIn  = np.where(np.abs(y)<=delta)[0]
        indsOut = np.where(np.abs(y)>delta)[0]

        keys = list(self.featureList.keys())
        # plt.figure(figsize=(5,5))
        for axIdx in range(len(self.featureList)):
            ax = plt.figure().add_subplot()
            # ax = plt.subplot(5,5,axIdx+1)

            if plotType.lower()=='scatter':
                ax.plot(data.loc[data.index[indsIn],keys[axIdx]],y.loc[y.index[indsIn]],'g.')
                ax.plot(data.loc[data.index[indsOut],keys[axIdx]],y.loc[y.index[indsOut]],'r.')
                plt.xlabel(keys[axIdx])
                plt.ylabel(target)
            elif np.isin(plotType.lower(),['hist','histogram']):
                h,bins = np.histogram(data.loc[data.index[indsIn],keys[axIdx]])
                ax.plot(bins[1:],h/np.sum(h))
                h,bins = np.histogram(data.loc[data.index[indsOut],keys[axIdx]])
                ax.plot(bins[1:],h/np.sum(h))
            elif plotType.lower()=='cumulative':
                h,bins = np.histogram(data.loc[data.index[indsIn],keys[axIdx]])
                ax.plot(bins[1:],np.cumsum(h/np.sum(h)))
                h,bins = np.histogram(data.loc[data.index[indsOut],keys[axIdx]])
                ax.plot(bins[1:],np.cumsum(h/np.sum(h)))

    def TestDistributionSimilarity(self,data1,data2,alpha=0.05):
        """
            Perform two-sided Kolmogorov Smirnoff test to examine
            if two samples are drawn from the same continuous distribution.
            The samples used are those with |Delta|<=delta and |Delta|>delta,
            where Delta is either the spherical equivalent delta, the cylinder or sphere delta.
            If the test statistics is small or the p-value high, we cannot reject the
            null hypothesis that the two samples are drawn from the same distribution. \n

            Parameters:
            ----------
            target: {'seDelta', 'cylnderDelta', 'sphereDelta'}, default: 'seDelta
                The Delta set, spherical equivalent delta, sphere or cylinder delta
            delta: float, default=0.25
                a positive scalar defining the Delta set
            eye: {'left','right','both'}, default='both'
                data to use
            alpha: float, default=0.05
                the confidence level of the KS test is: 1-alpha

            Output:
            --------
            res:
                a DataFrame including test statistics, p-value, and decision for the null hypothesis
        """
        if alpha>1 or alpha<0:
            raise Exception('Non valid alpha values. alpha must be in the range (0,1).')


        # check that data1 and data2 have similar field names
        if (data1.keys()==data2.keys()).all() ==False:
            raise Exception('The two datasets must contain similar field names')

        keys = data1.keys()
        n    = len(data1)
        m    = len(data2)
        res  = pd.DataFrame()
        for featIdx in range(len(keys)):
            # perform a k-s test for the similarity between the in and out groups
            k = stats.ks_2samp(data1.loc[:,keys[featIdx]],data2.loc[:,keys[featIdx]])
            res.loc[featIdx,'Feature']  = keys[featIdx]
            res.loc[featIdx,'KS']       = k.statistic
            res.loc[featIdx,'pvalue']   = k.pvalue
            res.loc[featIdx,'rejectH0'] = k.statistic>=(np.sqrt(-0.5*np.log(alpha/2))*np.sqrt((n+m)/(n*m)))
            res.loc[featIdx,'m']        = m
            res.loc[featIdx,'n']        = n

        return res

    def SelectFeaturesByKSTest(self,featureMat,labels,alpha=0.05,rejProp=0.5):
        """
            Select features automatically according to those that show high level of
            seperation between distributions of |Delta|<=delta
            and |Delta|>delta, based on the Kolmogorov-Smirnoff two-sided test statistics
            The KS statistic's null hypothesis is that the two set are from the same
            continuous distribution.

            Parameters:
            -----------
            featureMat: DataFrame
                feature matrix as dataFrame, with keys corresponding to features
            target : {'seDelta', 'cylinderDelta', 'sphereDelta'}, default='seDelta'
                the Delta set: spherical equivalent delta, sphere or cylinder delta
            delta  : float, default=0.25
                positive scalar defining the set |Delta|<=delta
            eye : {'left', 'right', 'both'} , default='both'
                data to use
            alpha :  float, default=0.05
                defines the KS confidence interval as confidence = 1-alpha
            prop: float, default0.5
                in pairwise comparison, a feature is selected if the null hypothesis
                was rejected at least round(prop*(numClasses-1)) times
                prop must be strictly positive and smaller than 1
            Output:
            -------
            featureMat: DataFrame
                Dataframe with keys corresponding to selected features
        """

        # Check that the number of labels matches number of observtions
        if len(featureMat)!=len(labels):
            raise Exception('Number of rows in featureMat must match that in labels vector')

        # Get all unique labels
        uLabels    = np.unique(labels)
        numClasses = len(uLabels)
        labelInds  = list(range(numClasses))
        # get all unique labels pairs
        labelPairs = list(itertools.product(labelInds,repeat=2))
        indsList   = np.ndarray(shape=(numClasses,numClasses),dtype=list)
        for lIdx in labelPairs:
            # print('Comparing classes {} and {}'.format(lIdx[0],lIdx[1]))
            inds1 = featureMat.index[np.where(labels==lIdx[0])[0]]
            inds2 = featureMat.index[np.where(labels==lIdx[1])[0]]
            data1 = featureMat.loc[inds1]
            data2 = featureMat.loc[inds2]
            res   = self.TestDistributionSimilarity(data1,data2,alpha=alpha)
            indsList[lIdx[0],lIdx[1]]= res.index[np.where(res['rejectH0']==True)[0]]

        numFeatures = len(featureMat.keys())
        indScore    = np.zeros(shape=numFeatures,dtype=np.int)
        for i1Idx in range(numClasses):
            for i2Idx in range(numClasses):
                for fIdx in indsList[i1Idx,i2Idx]:
                    indScore[fIdx]+=1
        indScore    = indScore/2

        featureInds = np.where(indScore>=np.round(rejProp*(numClasses-1)))[0]
        featureMat  = featureMat.loc[:,featureMat.keys()[featureInds]]
        self.featuresUsed = list(self.featureList.keys())[featureInds]
        return featureMat

    def SelectFeatures(self,fMat,labels,mode='k_best',alpha=0.05):
        '''
            Select features based on alpha threshold
        '''
        if len(labels)!=len(fMat):
            raise Exception('The number labels must match the number of rows in featureMat')

        # Normalize features between 0 and 1
        #        fMat = featureMat.copy()
        for k in fMat.keys():
            fMat[k] = (fMat[k] -np.min(fMat[k]))/(np.max(fMat[k])- np.min(fMat[k]))
        selector = sklearn.feature_selection.GenericUnivariateSelect(mode=mode)
        #        selector = sklearn.feature_selection.SelectFdr(score_func= sklearn.feature_selection.chi2,alpha=alpha)
        #        selector = sklearn.feature_selection.SelectFpr(score_func= sklearn.feature_selection.chi2,alpha=alpha)
        #        selector = sklearn.feature_selection.SelectKBest(score_func= sklearn.feature_selection.chi2,k=5)
        fit      = selector.fit(fMat,labels)
        featInds = np.where(fit.get_support())[0]
        self.featuresUsed = fMat.keys()[featInds]
        print('Num features selected : {}'.format(len(self.featuresUsed)))
        fMat     = fMat.loc[:,self.featuresUsed]
        return fMat

    @staticmethod
    def _ParseGender(genStr):
        if genStr.__class__==str:
            if genStr in ['f','F']:
                return 1
            elif genStr in ['m','M']:
                return 0
            else:
                return np.nan


