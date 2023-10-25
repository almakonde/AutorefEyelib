import pandas as pd
import numpy as np
from autorefeyelib.Refraction import vx120Transformer
from autorefeyelib.Refraction import vx120Imputer
from sklearn.impute import KNNImputer
class Loader:

    def __init__(self):
        self.transformer = vx120Transformer.Transformer()
        self.dataParsed  = False
        self.dataLoaded  = False

    def Load(self,fileName):
        """
            Load a joint VX-EMR database

            Parameters:
            -----------
            fileName: str
                a full path to a csv EMR-VX120 joint database
        """
        if not isinstance(fileName,str):
            raise ValueError('fileName must be a string. Got {fileName.__class__}')

        with open(fileName) as csvFile:
            self.data = pd.read_csv(csvFile,low_memory=False)
        self.dataLoaded = True
        self.dataParsed = False

        self.Left  = pd.DataFrame()
        self.Right = pd.DataFrame()
        self.Both  = pd.DataFrame()

    def Parse(self,vertexDistance=0.012,inclusion_criteria={}, impute_method='median',sphDelta=0.25,cylDelta=0.25):
        """
            Parse a merged (joint) DB of the vx120 and its matched EMR entries

            Parameters:
            -----------
            inclusion_criteria, optional, default ={}
             a dictionary with inclusion criteria
             list the name of the variable and the minimum maximum values
             e.g. {'Age':[15,80],'SphereDelta':[-1,1]}
            vertexDistance, float, optional, default=0.012
             vertex distance (m)
            impute_method, str, optional, default='median'
             how to treat missing values. options: 'median' (default)
             'knn'- impute with k-nearest neighbors, defult num neighbors = 5
             None- no imputation, keep data as is
            sphDelta/cylDelta, float, ddefault=0.25
             round sphere, cylinder to nearest sphDelta, cylDelta, respectively

             Returns:
             ---------

        """
        if not self.dataParsed:

            # impute missing values using the median
            if isinstance(impute_method,str):
                if impute_method.lower()== 'median':
                    # replace infinite values with column median
                    self.data.replace(np.inf,inplace=True)
                    # replace na values with column median
                    self.data.fillna(self.data.median(axis=0),inplace=True)

                elif impute_method.lower()=='knn': # Experimental feature
                    knn = KNNImputer()
                    # find only numeric fields
                    numeric_fields = (self.data.dtypes=='float64')|(self.data.dtypes=='int64')
                    self.data = knn.fit_transform(self.data[self.data.keys()[numeric_fields.values]],weights='distance')
            elif impute_method is not None:
                raise ValueError(f'impute_method must be string: median, knn or None, instead got {impute_method.__class__}')

            print(f'[Input] imputing missing and NA values using {impute_method} method')

            print('[Input] Transforming data')
            self.data = self.transformer.Transform(self.data,vertexDistance=vertexDistance)

            # computing additional features
            for eIdx in ['_Left','_Right']:
                # ACD to pupil radius
                self.data[f'AcdToPupilRadius_Ratio{eIdx}'] = 0.5*self.data[f'Pachy_MEASURE_Acd{eIdx}']\
                                                                /self.data[f'WF_RT_Fv_Meso_PupilRadius{eIdx}']
                # pupil radius to wtw ratio
                self.data[f'PupilRadiusToW2W_Ratio{eIdx}'] = 2*self.data[f'WF_RT_Fv_Meso_PupilRadius{eIdx}']\
                                                            /self.data[f'Pachy_MEASURE_WhiteToWhite{eIdx}']
                self.data[f'kRatio{eIdx}']                 = self.data[f'Topo_Sim_K_K1{eIdx}']/self.data[f'Topo_Sim_K_K2{eIdx}']


            for fIdx in self.data.keys():
                if fIdx.find('_Left')!=-1:
                    self.Left.loc[:,fIdx.replace('_Left','')] = self.data.loc[:,fIdx]
                elif fIdx.find('_Right')!=-1:
                    self.Right.loc[:,fIdx.replace('_Right','')] = self.data.loc[:,fIdx]
                else:
                    self.Left.loc[:,fIdx]  = self.data.loc[:,fIdx]
                    self.Right.loc[:,fIdx] = self.data.loc[:,fIdx]
            print('[Input] dividing data to Left and Right eye')

            validIndsL = np.ones(shape=len(self.Left),dtype=np.bool)
            validIndsR = np.ones(shape=len(self.Right),dtype=np.bool)
            rem_left   = 0 # number of invalid removed
            rem_right  = 0 # number of invalid removed
            for dk,dv in inclusion_criteria.items():
                if len(dv)>0:
                    if (dk in self.Left.keys()) and (dk in self.Right.keys()):
                        newIndsL = (self.Left[dk]<=max(dv))&(self.Left[dk]>=min(dv))
                        newIndsR = (self.Right[dk]<=max(dv))&(self.Right[dk]>=min(dv))

                        validIndsL = validIndsL&newIndsL
                        validIndsR = validIndsR&newIndsR
                        # invalid indices
                        l_invalid_sum = (newIndsL==False).sum()
                        r_invalid_sum = (newIndsR==False).sum()
                        rem_left += l_invalid_sum
                        rem_right+= r_invalid_sum
                        print(f'{dk} valid left : {newIndsL.sum()}, invalid {l_invalid_sum}')
                        print(f'{dk} valid right: {newIndsR.sum()}, invalid {r_invalid_sum}')
            print(f'Total removed left {rem_left}')
            print(f'Total removed right {rem_right}')
            self.Left  = self.Left.loc[validIndsL]
            self.Right = self.Right.loc[validIndsR]

            # Add subjective values to Left and Right databases
            self.Left['EMR:VisualAcuitySphere']    = self.Left['EMR:VisualAcuitySphere'].apply(self.transformer.Round,args=[sphDelta])
            self.Left['EMR:VisualAcuityCylinder']  = self.Left['EMR:VisualAcuityCylinder'].apply(self.transformer.Round,args=[cylDelta])
            self.Left['EMR:VisualAcuityAxis']      = self.Left['EMR:VisualAcuityAxis'].apply(self.transformer.Round,args=[1])
            self.Right['EMR:VisualAcuitySphere']   = self.Right['EMR:VisualAcuitySphere'].apply(self.transformer.Round,args=[sphDelta])
            self.Right['EMR:VisualAcuityCylinder'] = self.Right['EMR:VisualAcuityCylinder'].apply(self.transformer.Round,args=[cylDelta])
            self.Right['EMR:VisualAcuityAxis']     = self.Right['EMR:VisualAcuityAxis'].apply(self.transformer.Round,args=[1])

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

            # Compute subjective spherical equivalent
            self.Left['SubjectiveSphericalEquivalent']  = (self.Left['EMR:VisualAcuitySphere'] +
                                                           self.Left['EMR:VisualAcuityCylinder']/2)

            self.Right['SubjectiveSphericalEquivalent'] = (self.Right['EMR:VisualAcuitySphere'] +
                                                           self.Right['EMR:VisualAcuityCylinder']/2)

            # Compute spherical equivalent delta
            self.Left['SphericalEqDelta']  =  (self.Left['SphericalEquivalent_3'] -
                                               self.Left['SubjectiveSphericalEquivalent'])
            self.Right['SphericalEqDelta'] =  (self.Right['SphericalEquivalent_3'] -
                                               self.Right['SubjectiveSphericalEquivalent'])

            # Rearrange indices =
            self.Left.index  = range(len(self.Left))
            self.Right.index = range(len(self.Right))

            # Append for Both eyes
            self.Both = self.Left.append(self.Right)
            # Rearrange the indices for Both
            self.Both.index = range(len(self.Both))

            self.dataParsed = True

    @staticmethod
    def _AssignLabel(val,classes,groupEndBins=True):
        '''
         Assign labels from classes to a values
         val, float
          the delta between objective and subjective (sphere or cylinder)
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
                    return np.nan
        else:
            try:
                return classes.index(val)
            except:
                return np.nan