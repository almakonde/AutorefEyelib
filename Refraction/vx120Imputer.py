import os
import pandas as pd
import numpy as np
class Imputer:

    def __init__(self,vxDbPath = os.path.join(os.path.dirname(__file__),'data','vx130Data_Prevot.csv')):
        """ Load the database"""
        # print('[Info][VX120Imputer] Loading vx DB')
        # vxDB = pd.read_csv(vxDbPath,sep =",",
        #                    delimiter=",",
        #                    low_memory=False,
        #                    skip_blank_lines=True)

        self.data = pd.DataFrame()#vxDB

    def CompleteMissingFields(self,data):
        """
            Add missing fields and set to Nan
            Parameters:
            ----------
            data, DataFrame
                parsed dataframe of vx120 measurements
        """
        for kIdx in self.data.keys():
            if kIdx not in data.keys():
                data[kIdx] = np.nan
        return data

    def ImputeDF(self,data,strategy="median",imputeValues=[-1000,np.nan]):
        """
         Impute all the fields of a certain input dataframe using the same strategy
        """





        # locate all values to impute
        missing = data.replace(to_replace=[-1000],value=False)
        failed  = pd.DataFrame(index=data.index)
        print("[vx120Imputer] Imputing DF")
        if data.__class__==pd.DataFrame:
            # make sure the data is organized as in the db
            for kIdx in data.keys():
                if kIdx in self.data.keys():
                    dt            = np.asanyarray(list(data[kIdx].apply(self._ImputeCol,args=[self.data[kIdx],strategy])))
                    data[kIdx]    = dt[:,0]
                    missing[kIdx] = dt[:,1]
                    failed[kIdx]  = dt[:,2]

            for kIdx in self.data.keys():
                if kIdx not in data.keys():
                    if strategy.lower()=='median':
                        if self.data.dtypes[kIdx].name=='object':
                            #TODO: impute string values
                            pass
                        elif self.data.dtypes[kIdx].name=='float64':
                            data[kIdx] = self.data[kIdx].median()
                    elif strategy.lower()=='mean':
                        data[kIdx] = self.data[kIdx].mean().values


        return data

    @staticmethod
    def _ImputeCol(val,db_col_vals,strategy):
        if pd.isna(val):
            if strategy.lower()=='median':
                try:
                    val = db_col_vals.median()
                    return val, True, False
                except:
                    return np.nan, True, False
            elif strategy.lower()=='mean':
                try:
                    val = db_col_vals.mean()
                    return val,True, False
                except:
                    return np.nan, True, False
            else:
                raise(Exception(f'unknown option {strategy}'))
        elif val==-1000:
            if strategy.lower()=='median':
                try:
                    val = db_col_vals.meadian()
                    return val,False, True
                except:
                    return np.nan, False, True
            elif strategy.lower()=='mean':
                try:
                    val = db_col_vals.mean()
                    return val, False,True
                except:
                    return np.nan, False,True
            else:
                raise(Exception(f'unknown option {strategy}'))
        else:
            return val,False, False

    def Impute(self,fieldName,strategy="median"):
        """
         Complete missing or NA value
        """
        # get all valid indices in the column
        if fieldName.__class__==str:
            if strategy=="median":
                try:
                    val = self.data[fieldName].median()
                except:
                    val = pd.NA
            elif strategy=="mean":
                try:
                    val = self.data[fieldName].mean()
                except:
                    val = pd.NA
            return val
        else:
            return pd.NA
