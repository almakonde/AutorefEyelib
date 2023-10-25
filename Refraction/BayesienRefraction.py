import numpy as np
from scipy.ndimage import median_filter
from autorefeyelib.Refraction import vx120Transformer
from autorefeyelib.Refraction import vx120Imputer

class Classifier:
    def __init__(self):
        self.transformer = vx120Transformer.Transformer()
        self.cls         = []

    def Fit(self,featureMat,deltas,deltaRange,smooth_epdf=True,smoothing_sigma=1,med_filt_size=5):
        '''
         construct an empirical conditional probability for each delta in deltaRange
         to draw a value of the features in featureMat
         Parameters:
         -------
         featureMat, DataFrame
           feature names (keys) and values in rows
         deltas, DataFrame,
           values of the the deltas for each feature set
         deltaRange,
           valid delta values to include in the classifier and for which empirical probabilities are computed

         Output:
         -------
         prob, dictionary
          a dictionary with empirical conditional probabilities
          of size 2*len(featureMat) keys
          of featureName_val and conditional empirical probability

        '''
        model = {}

        featureMat = self._BinFeatureMatValues(featureMat)
        feature_importance = {}
        lims = [0,0.25,0.5]
        for kIdx in list(featureMat.keys()):
            print(f'Fitting {kIdx}')
            model[kIdx], model[f'{kIdx}_vals'],model[f'{kIdx}_exp'] = self.conditional_delta_epdf( featureMat[kIdx].values,
                                                                                                deltas.values,deltaRange,
                                                                                                smooth=smooth_epdf,
                                                                                                smoothing_sigma=smoothing_sigma,
                                                                                                med_filt_size=med_filt_size)

            # score according to the expectation value from conditional probability function

            feature_importance[kIdx] = np.zeros(3) # initialize
            for fIdx in range(len(featureMat[kIdx])):
                # find the index corresponding to the feature value
                ind = np.abs(featureMat.loc[featureMat.index[fIdx],kIdx]-model[f'{kIdx}_vals']).argmin()
                for dIdx in range(len(lims)):
                    if np.abs(deltaRange[int(model[f'{kIdx}_exp'][ind])]-deltas.iloc[fIdx])<=lims[dIdx]:
                        feature_importance[kIdx][dIdx]+=1
            feature_importance[kIdx]/=len(featureMat)



        self.cls = model
        self.feature_importance = feature_importance

    @ staticmethod
    def ecdf(vals,smooth=False):
        uVals = np.sort(np.unique(vals))
        prob  = np.zeros(len(uVals))
        N     = len(vals)
        for uIdx in range(len(uVals)):
            prob[uIdx] = np.sum(vals<=uVals[uIdx])/N
        if smooth:
           prob  = median_filter(prob,3)
        return prob, uVals

    @staticmethod
    def epdf(vals, bins=None, smooth=False):
        if bins is not None:
            uVals = bins
        else:
            uVals = np.sort(np.unique(vals))
        prob  = np.zeros(len(uVals))
        for uIdx in range(len(uVals)):
            prob[uIdx] = np.sum(vals==uVals[uIdx])/len(vals)

        if smooth:
            prob = median_filter(prob,3)
            prob/=np.trapz(prob,x=uVals)
        return prob, uVals

    def conditional_delta_epdf(self,vals,delta, deltaBins,smooth=False,smoothing_sigma=1,med_filt_size=5):
        '''
         Parameters:
         -----------
          deltaBins are unique bins of delta for which to create the epdf
          delta and vals must have the same number of rows
          deltaBins is a sorted list of bins in delta for which to compute the epdf
         output:
         -------
          epdf is a len(unique(vals)) by len(deltaBins) array of empirical conditional probabilities of
          drawing a delta value given a feature value
          vals,
            unique values of the epdf
        '''
        v             = np.sort(np.unique(vals))
        prob_f,_      = self.epdf(vals,bins=v,smooth=smooth) # epdf of the feature in vals
        prob_delta, _ = self.epdf(delta,bins=deltaBins,smooth=smooth) # epdf of the delta
        e             = np.zeros((len(deltaBins),len(v)))
        expectation   = np.zeros(len(v))

        if smooth:
            sKernel = np.exp(-(np.arange(-1,1,0.5)**2)/(2*smoothing_sigma**2))/np.sqrt(2*np.pi*smoothing_sigma)
            sKernel/= np.sum(sKernel)


        for dIdx in range(len(deltaBins)):
            ff                = vals[delta==deltaBins[dIdx]]
            prob_f_delta,_    = self.epdf(ff,v)
            e[dIdx]           = prob_f_delta*prob_delta[dIdx]/prob_f

            # S = np.sum(e[dIdx])
            # if S>0:
            #     e[dIdx]/=S
            if smooth:
                e[dIdx] = median_filter(e[dIdx],med_filt_size)
                c       = np.convolve(e[dIdx],sKernel,mode='same')
                if len(c)==len(deltaBins):
                    e[dIdx] = c
            #     S = np.sum(e[dIdx])
            #     if S>0:
            #         e[dIdx]/=S
        for vIdx in range(len(v)):
            try:
                expectation[vIdx] = deltaBins.index(np.round(4*np.dot(e[:,vIdx],deltaBins))/4)
            except:
                expectation[vIdx] = e[:,vIdx].argmax()

        return e.T, v,expectation

    def _BinFeatureMatValues(self,fMat):
        rounding = {
                               'Age':1,
                               'Gender':1,
                               'Topo_Sim_K_K1':0.25,                           # K1
                               'Topo_Sim_K_K2':0.25,                           # K2
                               'kRatio':0.05,                                  # computed
                               'WF_SPHERE_R_3':0.25,                       # sphere
                               'WF_CYLINDER_R_3':0.25,                     # cylinder
                               'WF_AXIS_R_3':1,                             # Axis (radians)
                               'WF_RT_Fv_Zernike_Photo_di_Z_2_-2':0.02,    # astigmatism
                               'WF_RT_Fv_Zernike_Photo_di_Z_4_0':0.02, # spherical aberration
                               'WF_RT_Fv_Zernike_Photo_di_Z_3_1':0.02, # primary coma photo
                               'WF_RT_Fv_Meso_PupilRadius':0.1,               # pupil radius
                               'WF_RT_Fv_Zernike_Meso_di_Z_2_-2':0.02,     # astigmatism
                               'WF_RT_Fv_Zernike_Meso_di_Z_4_0':0.02, # spherical abberation diopter
                               'WF_RT_Fv_Zernike_Meso_di_Z_3_1':0.02,  # primary coma meso
                               'Pachy_MEASURE_Acd':0.05,                    # anterior chamber depth
                               'Pachy_MEASURE_WhiteToWhite':0.1,          # white to white
                               'Pachy_MEASURE_KappaAngle':1,            # kappa angle
                               'Pachy_MEASURE_Thickness':10,          # pachimetry
                               'Tono_MEASURE_Average':1,                    # tonometry
                               'Topo_KERATOCONUS_Kpi':10,                    # keratoconus index
                               'AnteriorChamberVol':5,                      # computed
                               'AcdToPupilRadius_Ratio':0.05,                  # computed
                               'PupilRadiusToW2W_Ratio':0.05,                  # computed
                               'J0_3':0.2,                                      # computed
                               'J45_3':0.2,                                     # computed
                               'BlurStrength_3':0.2                             # computed
                                }
        for kIdx in fMat.keys():
            if kIdx in list(rounding.keys()):
                fMat[kIdx] = fMat[kIdx].apply(self.transformer.Round,args=[rounding[kIdx]])
            else:
                fMat.drop(columns=kIdx,inplace=True)
        return fMat

    def Predict(self,features):
        '''
         Parameters
         ----------
         features, DataFrame or Series with
           n features (keys) and corresponding values

        '''
        # 1. generate the empirical conditional probability
        deltaProb   = []
        deltaChoice = []
        expChoice   = []
        if (len(self.cls)>0)&(isinstance(self.cls,dict)):
            deltaProb = np.ones(self.cls[list(self.cls.keys())[0]].shape[1])
            deltaChoice = []
            for kIdx in features.keys():
                if kIdx in list(self.cls.keys()):
                    # find the row in the empirical pdf dictionary
                    row = np.abs(self.cls[f'{kIdx}_vals']-features[kIdx]).argmin()
                    # compute multiplicative probability
                    deltaProb*=self.cls[kIdx][row]
                    # record the delta corresponding to the peak of the emp. probability
                    deltaChoice.append(self.cls[kIdx][row].argmax())
                    expChoice.append(np.round(self.cls[f'{kIdx}_exp'][row]*4)/4)
                else:
                    print(f'{kIdx} is not in prob keys')
            S = np.sum(deltaProb)
            if S>0:
                deltaProb/=S
            # deltaProb = median_filter(deltaProb,10)
            # gKernel = np.exp(-(np.arange(-1,1,0.5)**2)/(2*1**2))/np.sqrt(2*np.pi*1)
            # gKernel/=np.sum(gKernel)
            # deltaProb  = np.convolve(deltaProb,gKernel,mode='same')

            # S = np.sum(deltaProb)
            # if S>0:
            #     deltaProb/=S
        else:
            print('Classifier is not yet initialized. Please run Fit() with appropriate data')

        return deltaProb,deltaChoice,expChoice

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