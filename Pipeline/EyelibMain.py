import os
import sys

import pandas as pd

from autorefeyelib.Refraction import Predictor as refractionPredictor
from autorefeyelib.Parsers import Revo
from autorefeyelib.Parsers import vx120
from autorefeyelib.Parsers import xmlParser
# from autorefeyelib.Refraction import vx120Imputer
from autorefeyelib.Refraction import vx120Transformer
from autorefeyelib.IOLpower import Predictor as iolPredictor
from autorefeyelib.Risks import Risks
from autorefeyelib.RefractiveSurgery import SurgeryRecommender

class Pipeline:
    """ Main pipeline for the Eyelib analysis module
        Rhe pipline runs the following algoerithms sequentially:
        vx120 transformer
        subjective refraction predictor
        Risk predictor
        Revo image parser
        IOL power predictor
        Surgery recommender

        Call by running  Run()
    """
    def __init__(self):
        self.vx120Transformer    = vx120Transformer.Transformer()
        self.vx120Parser         = vx120.Parser()
        self.refractionPredictor = refractionPredictor.Predictor()
        self.revoParser          = Revo.Parser()
        self.iolPredictor        = iolPredictor.Predictor()
        self.riskPredictor       = Risks.Predictor()
        self.surguryRecommender  = SurgeryRecommender.Recommender()
        self.vx40Parser          = xmlParser.Parser()

    def Run(self,vx120zipFile,revoImageFile,vx40File, virtualAgentFile=None,
            patID=None,vertexDistance=0.012,
            Aconst=[118,118.5,118.9,119.4],
            targetRefractionIOL=[0,-0.5,-1,-1.5],
            targetRefractionSurg=(0,0),
            pvd=(False,False),dominantEye=None,
            priorLasik=(False,False),flapThickness=(100,100)):

        """ Run parser and predictors for refraction, risks and IOL power


         Parameters
         ------------
          vx120zipFile : str
           a path to the vx120 zip file output including a single measurement
          revoImageFile : str
           a path to the revo bmp output file
          vx40File : str
            a path to the Visionix vx40 xml file
          virtualAgentfile : str
            a path to the virtual agent file
          patID : str, default=None
           patient ID to assign to the output DataFrame
           when patID=None, the patient ID from the vx120Parser will be assigned
          vertexDistance, float, default =0.012
           the vertex distance in mm
          Aconst, float, default=118.9
           IOl manufacturer A-constant
          targetRefractionIOL, array(float), default=[0]
           target refraction (D) for IOL implant
          targetRefractionSurg, tupple(float), default=(0,0)
           target refraction (D) for refrctive surgery
           for (right,left) eye, multiple values allowd,
           the algorithm then returns the IOL power predicted
           for each target refraction value
          pvd : tuple, bool, default=(False,False)
           partial vitreous detachment for (Right, Left) eye
          dominantEye : str , default='right'
           dominant eye options right/left
          priorLasik : tuple (bool,bool), default=(False,False)
            if Lasik was previously performed on (Right, Left) eye
          flapThicknes : tuple (float, float), default = (100,100)
           flap thickness (mu m)

         Output
         -------
          predictedRef : DataFrame
            predicted values for subjective sphere cylider, axis and contact lenses
          risks : DataFrame
            predicted risks, pachy, tono, angle closure
          iolPredList : list(DataFrame)
            predicted iol power and type for each value of the A-constant  and target refraction
          surgeryRecommendation, dict
            recommended surgery (dict) for right and left eye, including
            risk of ectasia, dominant eye and parameters used for recommendation
          dataOut, DataFrame
           a dataframe containing all measurements parsed and transformed fields
           from the vx120 and concatanated OCT data
        """
        if vx40File is not None:
          vx40Data   = self.vx40Parser.Parse(vx40File).T
        else:
          vx40Data = None


        # Parse vx120 data
        vx120Data  = self.vx120Parser.ParseVXFromZip(vx120zipFile)
        # transform data to output format
        self.vx120Transformer.Transform(vx120Data,vertexDistance=vertexDistance)

        # Set the index for the dataframe
        if patID is None:
            patID = vx120Data.index
        else:
            vx120Data.index = [patID]

        # Predict subjective refraction
        predictedRef = self.refractionPredictor.PredictSubjectiveRefraction(vx120Data,vx40Data, returnVals='predicted')
        # check predicted ref vs glasses and signal in case of large difference

        # Predict risks
        risks       = self.riskPredictor.Predict(vx120Data)
        risks.index = [patID]

        # Parse OCT data
        revoData    = self.revoParser.Parse(revoImageFile,output='row',patID=patID)
        # revoData.index = dataOut.index

        # combine output
        dataOut     = pd.concat([vx120Data,revoData],axis=1,ignore_index=False,verify_integrity=True,copy=True)

        # Run IOL-power predictor
        iolPredList = []
        for tIdx in targetRefractionIOL:
            iolPrediction = pd.DataFrame()
            for aIdx in Aconst:
                # print(f'IOL prediction Aconst = {aIdx} target refraction {tIdx}')
                # Predict the correction to the IOL power and then the power
                pred          = self.iolPredictor.PredictdP(aIdx,dataOut,targetRefraction=tIdx,pDelta=0.5,rDelta=0.25)
                iolPrediction = pd.concat([iolPrediction,pred],axis=0)
            iolPredList.append(iolPrediction)
        # use characteristic values + objective measurements to recommend a surgery
        # reqCorD = ((targetCorRefSurg[0]-predictedRef['Predicted_Sphere_Right'].values[0]),
        #            (targetCorRefSurg[1]-predictedRef['Predicted_Sphere_Left'].values[0]))

        surgeryRecommendation = self.surguryRecommender.RunFromParsedData(dataOut,predictedRef,targetRefractionSurg,
                                                                        pvd=pvd,priorLasik=priorLasik,
                                                                        dominantEye = dominantEye,
                                                                        flapThickness=flapThickness,
                                                                        vertex_distance=vertexDistance)

        return predictedRef, risks, iolPredList, surgeryRecommendation, dataOut