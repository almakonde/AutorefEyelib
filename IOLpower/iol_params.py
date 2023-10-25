"""
    Parameter file for the training and operation of the IOL power predictor.
    This file includes the class dev_params related only to development.
    Load parameteers by
    from autorefeyelib.IOLpower import iol_params as params

    Access to parameters is done by by dot notation, as attributed of params
"""

features    = ["age","meanK","ACD","WTW","axialLength","targetRefraction"]
valid_ranges= {"age":[40,100],
                "meanK":[35,47],
                "axialLength":[20,40],
                "ACD":[2,5],
                "WTW":[10,14],
                "targetRefraction":[-10,10],
                "deltaR":[-3,3],
                "Followup_Days":[10,100]}
formulas       = ["SRKT","Shammas","Haigis","Binkhorst-2","Holladay-1","Hoffer-Q","Olsen"]
features_reg   = ["age","axialLength","ACD","meanK","targetRefraction"]
features_class = ["age","axialLength","ACD","meanK","targetRefraction"]
pDelta         = 0.5
rDelta         = 0.25
Aconst         = 118.9
alpha          = [0.1]
n_c={"SRKT":1.333,
    "T2":1.333,
    "Barrett-1":1.336,
    "Hoffer-Q":1.336,
    "Olsen":1.3315,
    "Shammas":1.3333,
    "Haigis":1.332,
    "Holladay-1":1.336,
    "Binkhorst-2":1.336}
n_v={"SRKT":1.336,
    "T2":1.336,
    "Barrett-1":1.336,
    "Hoffer-Q":1.336,
    "Olsen":1.336,
    "Shammas":1.336,
    "Haigis":1.336,
    "Holladay-1":1.336,
    "Binkhorst-2":1.333}


class dev_params:
    """ Parameters related to development"""
    def __init__(self):
        self.paramsGrid = {"n_estimators":[50,60,70,80,90,100],
                    "criterion":["entropy","gini"],
                    "max_depth":[50,100,150,200,250,300],
                    "ccp_alpha":[1e-3, 1e-4, 1e-5, 1e-6]}
        self.frac_training       = 0.85
        self.num_exp             = 1
        self.showResultFigures   = False
        self.min_followup_days   = 10
        self.max_followup_days   = 120
        self.tune_hyperparameters= False
        self.export_model        = False
        self.reg_chooser_params = {'algorithm':'kd_tree','n_neighbors':20,'p':1,'leaf_size':10}

dev = dev_params()