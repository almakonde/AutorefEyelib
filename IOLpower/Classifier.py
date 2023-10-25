import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from autorefeyelib.IOLpower import Formulas
class Classifier:

    def __init__(self):
        # self._goal_function = lambda alpha, dRef, dP, maxRef,maxP: alpha*(abs(dRef)/abs(maxRef))+(1-alpha)*(abs(dP)/abs(maxP))
        self._goal_function = lambda dRef, dP, maxRef,maxP: (abs(dRef))/maxRef
        self.formulas = Formulas.Formulas()


    def Fit(self,Aconst,featureMat,labels,numExp=5,num_splits = 3, percTraining=0.9,alpha=0.4,classifierParams=None):
        """
            Train and validate a random forest classifier
            Parameters:
            -----------
            Aconst, float
                Manufacturer IOL A constant
            featureMat, DataFrame
                feature matrix
            labels, array(int)
              an array of ground truth labels the same length as featureMat
            numExp, int, default=5
                number of times to train a classifier
            percTraining, float
                percentage of the db (size of featureMat) on which
                to train. The remaining (1-percTraining) is used for validation
            alpha, float
                the ratio of refraction to power in the goal function
            classifierParams, dict
                dictionary with classifier paramters
                parameters which are not specified in classifierParams
                will be set to their default values
                set classifierParams=None for defaults

            Output:
            --------
            clf, RandomForestClassifier
                trained classifier, the best obtained in numExp
        """
        cParams = self._ParseInputClassifierParams(classifierParams)
        score   = np.zeros(numExp)
        Tind    = round(percTraining*len(featureMat))
        err     = np.zeros((len(featureMat[Tind:]),numExp)) # final refraction vs predicted
        pErr    = np.zeros((len(featureMat[Tind:]),numExp)) # power implanted error
        clf     = RandomForestClassifier()
        # clf     = LogisticRegression()
        # clf.set_params(**cParams)
        # clf     = LinearSVC(C=13,verbose=False,max_iter=15000,tol=1e-4,penalty='l2',multi_class='crammer_singer')


        bestScore      = 0
        bestClassifier = 0
        # if numExp>=2:
        #     skf     = StratifiedKFold(n_splits=numExp,shuffle=False)
        #     for trainIdx,testIdx in skf.split(featureMat,labels):
        #         clf.fit(featureMat.iloc[trainIdx].values,labels[trainIdx])
        #         # TODO: introduce new score function to inclufe the 0.25D range
        #         score.append(clf.score(featureMat.iloc[testIdx],labels[testIdx]))
        #         if score[-1]>bestScore:
        #             bestScore = score[-1]
        #             bestClassifier =
        # else:
        for eIdx in range(numExp):
            print(f'Training {eIdx+1}/{numExp}')
            shuffledInds = np.random.permutation(len(featureMat))
            trainInds    = shuffledInds[:Tind]
            validateInds = shuffledInds[Tind:]
            # Train the model
            clf.fit(featureMat.iloc[trainInds],labels[trainInds])
            score[eIdx]       = clf.score(featureMat.iloc[validateInds],labels[validateInds])
            if score[eIdx]>bestScore:
                bestScore      = score[eIdx]
                bestClassifier = eIdx
            # clf.n_estimators+=10

        # fImportance = clf.feature_importances_

        # compute the error for the used predicted IOL formula
        # pMean  = pd.DataFrame(abs(err)).mean(axis=1).dropna().mean()

        # compute the cumulative error
        # fCumul = pd.DataFrame()
        # pCumul = pd.DataFrame()
        # limF   = np.arange(0,0.75,0.125)
        # limP   = np.arange(0,8,0.125)
        # for lIdx in range(len(limF)):
        #     # refraction cumulative error
        #     fCumul.loc[lIdx,'Predicted']     = (abs(err[:,bestClassifier])<=limF[lIdx]).sum()/len(err[:,bestClassifier])
        #     # power cumulative error
        # for lIdx in range(len(limP)):
        #     pCumul.loc[lIdx,'Predicted']     = (abs(pErr[:,bestClassifier])<=limP[lIdx]).sum()/len(pErr[:,bestClassifier])


        # print(f'\n MAE-final refraction  Algo. {pMean:.2f}\n  Mean score:{score.mean():.2f}\n')
        # print(f'MAE-final refraction\n {fRefErr.abs().mean()}\n')
        # print(f'Accuracy ratio (err. formula)/(err. prediction) for best score: \n {ratio}')
        # print(f'Prediction score, best {score[bestClassifier]:.2f}, average:{score.mean():.2f}, min:{score.min():.2f}')

        return clf

    def _ParseInputClassifierParams(self,classifierParams):
        """
            Get default parameters and set input parameters
            Parameters:
            ------------
            classifierParams, dict
                dictioinary with keys as parameter names corresponding to parameters of the RF classifier
            Output:
            ---------
            cParams, dict
                 a dictionary with parameter names and values

        """

        cParams = self._GetDefaultClassifierParams()
        if classifierParams is not None:
            assert(isinstance(classifierParams,dict))
            for cIdx in classifierParams.keys():
                if cIdx in cParams.keys():
                    cParams[cIdx] = classifierParams[cIdx]
                else:
                    raise ValueError(f'The key {cIdx} does not exist in the classifier parameters. Please check spelling')

        return cParams

    @staticmethod
    def _GetDefaultClassifierParams():
        cParams      = {'n_estimators':100,#70,
                        'oob_score':True,
                        'criterion':'entropy',
                        'max_depth':250,#150,
                        'bootstrap':True,
                        'ccp_alpha':1e-6,#0.000001,
                        'max_leaf_nodes':None,
                        'max_features':'auto',
                        'class_weight': None,
                        'max_samples': None,
                        'min_impurity_decrease': 0.0,
                        'min_impurity_split': None,
                        'min_samples_leaf': 1,
                        'min_samples_split': 2,
                        'min_weight_fraction_leaf': 0.0,
                        'n_jobs': None,
                        'random_state': None,
                        'verbose': 0,
                        'warm_start': True}
        return cParams

    def GetClasses(self,formulas,Aconst,meanK,acd,wtw,axialLength,Rt,Pi, Rf, meanCornealHeight,meanACD,surgeonFactor,hofferQPersonalizedACD,pDelta=0.5,rDelta=0.25):
        """
          Assign classes to feature vector/matrix according to prediction of IOL formulas
          vs. implanted IOL power and final refraction.

          Parameters:
          -----------
            Aconst, float,
                manufacturere A constant
            meanK, float
                average keratometry (D)
            acd, float
                anterior chamber depth (mm)
            wtw, float
                white to white (mm)
            axialLength, float
                axial length (mm)
            Rf, array,
                final refraction (D)
            Rt, array
                target refraction (D)
            Pi, array
                power impolanted (D)

            Returns:
            -----------
            labels, array int
                an array of labels (class) the same length as the input data
            P, DataFrame
                Power computed by any one of the formulas for each data set
            R, DataFrame
                Predicted refraction for any power predicted
            elps, DataFrame
                Predicted Effective Lens Possition (mm) for each data set
            al, DataFrame
                Predicted (adjusted) axial length (mm) for each data set

        """
        # compute the refraction difference
        P, R,elps,al = self.formulas.ComputeAllFormulas(formulas,Aconst,meanK,acd,wtw,axialLength,Rt,meanCornealHeight,meanACD,surgeonFactor,hofferQPersonalizedACD,pDelta=pDelta)
        # Pp, _,_,_    = self.formulas.ComputeAllFormulas(formulas,Aconst,meanK,acd,wtw,axialLength,Rf,meanCornealHeight,meanACD,surgeonFactor,hofferQPersonalizedACD,pDelta=pDelta)
        # Aconst,meanK,acd,wtw,axialLength,Rt,meanCornealHeight,meanACD,surgeonFactor,hofferQPersonalizedACD
        if rDelta is not None:
            R = np.round(R/rDelta)*rDelta
        if pDelta is not None:
            P = np.round(P/pDelta)*pDelta

        # pRefErr = pd.DataFrame(columns=P.columns)
        fRefErr = pd.DataFrame(columns=R.columns)
        for pIdx in P.keys():
            # pRefErr[pIdx]  = (Pi  - P[pIdx]).abs()
            fRefErr[pIdx]  = (Rf  - R[pIdx]).abs()
        # maxF   = fRefErr.abs().max(axis=0)
        # maxP   = pRefErr.abs().max(axis=0)
        # scores = self._goal_function(alpha, fRefErr,pRefErr,maxF,maxP)


        labels = fRefErr.apply(np.argmin,axis=1)

        return labels, P, R, elps,al

    def _Fit(self,Aconst,age,meanK,acd,wtw,axialLength,Rt,labels,alpha=0.4):

        featureMat = pd.DataFrame()
        featureMat.loc[:,'age']              = age
        featureMat.loc[:,'meanK']            = meanK
        featureMat.loc[:,'ACD']              = acd
        featureMat.loc[:,'WTW']              = wtw
        featureMat.loc[:,'axialLength']      = axialLength
        featureMat.loc[:,'targetRefraction'] = Rt
        clf   = RandomForestClassifier(n_estimators=500,
                                        oob_score=True,
                                        criterion='entropy',
                                        max_depth=350,
                                        bootstrap=True,
                                        ccp_alpha=0,
                                        max_leaf_nodes=None,
                                        max_features='auto')
        # clf = LinearSVC()
        # get labels
        # labels = self.GetClasses(Aconst,meanK,acd,wtw,axialLength,Rt,Pi,Rf,alpha=alpha)
        clf.fit(featureMat,labels)
        return clf