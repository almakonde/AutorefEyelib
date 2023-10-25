import re
import cv2
import os
import pandas as pd
import numpy as np
import pytesseract
from pytesseract import Output

class Parser:
    """
         parse data from a fixed-format Revo bmp file output using OCR

    """
    def __init__(self):
        # Create a grid for the right and left eye sub-image data where text is expected to appear
        cols_r       = [int(190 +(rIdx/2)*110) if (rIdx%2)==0 else int(270+110*(rIdx-1)/2)  for rIdx in range(8)]
        cols_l       = [int(1020+(rIdx/2)*110) if (rIdx%2)==0 else int(1100+110*(rIdx-1)/2) for rIdx in range(8)]
        rows_r       = [int(255 + 25*(rIdx/2)) if (rIdx%2)==0 else int(275+25*(rIdx-1)/2)   for rIdx in range(22)]
        rows_l       = [int(255 + 25*(rIdx/2)) if (rIdx%2)==0 else int(275+25*(rIdx-1)/2)   for rIdx in range(22)]
        self.rows_r  = np.asanyarray(rows_r,dtype=np.int)
        self.cols_r  = np.asanyarray(cols_r,dtype=np.int)
        self.rows_l  = np.asanyarray(rows_l,dtype=np.int)
        self.cols_l  = np.asanyarray(cols_l,dtype=np.int)
        self.marks_r = [47,67]   # column boundaries (left right) for the  icon acceptable measurement icon (V)
        self.marks_l = [880,900] # column boundaries (left right) for the  icon acceptable measurement icon (V)
        self.boxWidht  = 80  # pix
        self.boxHeight = 20  # pix

    def Parse(self,imgFilePath,patID=0,output='row',gamma=0.9,minWidth=3,minHeight=10,maxArea=1500,minConf=5,exportImage=False, exportImageName='revo_results.bmp'):
        """
             Parse OCT data from a fixed format report of Optopol
             Parameters:
             ----------
             imgFilePath, str
               path to the report bmp file
               image must be 1485X1050 pix in size
             patID, object, default=0
               the patID to assign to the row output (output='row')
               patID will not be applied to the table in the case output='table'
             output, str
              output type: 'row'   - one DF row
                           'table' - DF with measurments 1-10 and Avg for Right/Left
             gamma, float, default = 0.8
               image stretchig factor
             minWidht, int, default = 3
              minimal width of the bounding box of a word
             minHeight, int, default=10
                minimal height of the bounding box of a word
             maxArea, int, default = 1500
               maxiaml area of the bounding box of a word
             minConf, int, default=5
               minimal confidence of a word (0-100)
             exportImage, bool, default=False
               flag to export the resulting image with text bounding boxes
             exportImageName, str, default=revo_results.bmp
             name of the exported images

             Output:
             -------
             res, dataFrame
              results dataframe for left and right eye
        """
        if imgFilePath is None:
            if output.lower()=='row':
                res = pd.DataFrame(index=[patID])
                fName = ['AxialLength','ACD','LT','CCT']
                inds  = [0,1,2,3,4,5,6,7,8,9,'Avg']
                for lIdx in ['_Right','_Left']:
                    for fIdx in fName:
                        for iIdx in inds:
                            res.loc[patID,f'{fIdx}_{iIdx}{lIdx}'] = None
            elif output.lower()=='table':
                res = pd.DataFrame(index =[0,1,2,3,4,5,6,7,8,9,'Avg'],columns=['AxialLength','ACD','LT','CCT'])
            else:
                raise ValueError(f'Unknown option output={output}')
            return res
        if not os.path.exists(imgFilePath):
            Warning(f"Cannot find the file {imgFilePath}. Returning nan values")
        else:
            # Read image
            img  = cv2.imread(imgFilePath)

        if not imgFilePath.__class__==str:
            raise ValueError(f"imgFilePath must be a string. Got {imgFilePath.__class__}")
        if not (img.shape[1]==1485)&(img.shape[0]==1050): #NOTE: consider removing (too restrictive)
            raise ValueError(f"Image must be (1050x1485). Got img.shape={img.shape}")
        # check input values
        if not (minWidth.__class__==int)&(minWidth>1)&(minWidth<img.shape[1]):
            raise ValueError(f"minWidth must be a positive integer smaller than image width. Got {minWidth} but num. columns= {img.shape[1]}")
        if not (minHeight.__class__==int)&(minHeight>1)&(minHeight<img.shape[0]):
            raise ValueError(f"minHeight must be a positive integer smaller than image height. Got {minHeight} but num. rows={img.shape[0]}")
        if not (maxArea>1)&(maxArea<img.shape[0]*img.shape[1]):
            raise ValueError(f"maxArea must be smaller than the image total number of pixels. Got maxArea={maxArea}, but img shape={img.shape}")
        if not (minConf>0)&(minConf<100):
            if minConf<0:
                newMinConf = 0
            elif minConf>100:
                newMinConf=100
            Warning(f"minConf must be in the range (0,100). Got {minConf}. setting min conf to {newMinConf}")
            minConf = newMinConf
        if not (output.lower() in ['row','table']):
            raise ValueError(f"variable 'output' must be either row or table. Got {output}")
        if not (exportImage.__class__==bool):
            raise ValueError(f"exportImage variable must be a boolean flag. Got {exportImage}")
        if not (exportImageName.__class__==str):
            raise ValueError(f"exportImageName must be a valid string. Got {exportImageName}")

        ims  = np.asanyarray(np.asanyarray(img,dtype=np.float)**gamma,dtype=np.uint8)
        gray = cv2.cvtColor(ims, cv2.COLOR_RGB2GRAY)
        imc  = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        data_l,res_l = self._ParseImage(imc,'left',
                                        minWidth=minWidth,
                                        minHeight=minHeight,
                                        maxArea=maxArea,
                                        minConf=minConf)
        data_r,res_r = self._ParseImage(imc,'right',
                                        minWidth=minWidth,
                                        minHeight=minHeight,
                                        maxArea=maxArea,
                                        minConf=minConf)

        if exportImage:
            for i in range(len(data_r['text'])):
                (x, y, w, h) = (data_r['left'][i], data_r['top'][i], data_r['width'][i], data_r['height'][i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            for i in range(len(data_l['text'])):
                (x, y, w, h) = (data_l['left'][i], data_l['top'][i], data_l['width'][i], data_l['height'][i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.imwrite(exportImageName,img) # debug image

        if output.lower()=='table':
            for cIdx in res_r.columns:
                res_r.rename(columns={cIdx: cIdx+'_Right'}, inplace=True, errors='raise')
            for cIdx in res_l.columns:
                res_l.rename(columns={cIdx:cIdx+'_Left'},inplace=True, errors='raise')
            res = pd.concat([res_r,res_l],axis=1,ignore_index=False,verify_integrity=True)
        elif output.lower()=='row':
            res = pd.DataFrame(index=[patID])
            for cIdx in res_l.columns:
                for idx in res_l.index:
                    res.loc[patID,f'{cIdx}_{idx}_Left']  = res_l.loc[idx,cIdx]
            for cIdx in res_r.columns:
                for idx in res_r.index:
                    res.loc[patID,f'{cIdx}_{idx}_Right'] = res_r.loc[idx,cIdx]

        return res

    def _GetCombinedBlocksImage(self,imc,eye):
        """
         crop the input image and rebuild it without the grid
         Parameters:
         ----------
          imc, 2d array (float/uint8)
           input image
          eye, str
           eye indicator, 'left' or 'right'
         Output:
         -----
         imTotal, array 2d
          image, with a similar size as the input image
          for which only the text boxes appear
          text boxes coordinates are defined in self.rows_r(l)/self.cols_r(l)

        """
        if not (eye.__class__==str)&(eye in ['left','right']):
            raise ValueError("eye variable must be either 'left' or 'right'. Got {eye}")
        imTotal = np.ones(imc.shape,dtype=np.uint8)*255
        for rIdx in np.arange(0,len(self.rows_r)-1,2):
            for cIdx in np.arange(0,len(self.cols_r)-1,2):
                if eye.lower()=='right':
                    r0 = self.rows_r[rIdx]
                    r1 = self.rows_r[rIdx+1]
                    c0 = self.cols_r[cIdx]
                    c1 = self.cols_r[cIdx+1]
                elif eye.lower()=='left':
                    r0 = self.rows_l[rIdx]
                    r1 = self.rows_l[rIdx+1]
                    c0 = self.cols_l[cIdx]
                    c1 = self.cols_l[cIdx+1]
                imTotal[r0:r1,c0:c1] =imc[r0:r1,c0:c1]

        return imTotal

    def _ParseImage(self,img,eye,minWidth=3,minHeight=10,maxArea=1500,minConf=5,minBlobAreaPix=4,min_v_area=30):
        """
             Run OCR on the output image of the octopol OCT
             Parameters:
             -----------
             img, input b&w/grayscale or 2d array
             eye : str
               left or right eye image, eye='left'/eye='right'
             minwidth : int
               minimal width of the bounding box of the word
             minHeight : int
                minimal height of the bounding box of the word
             maxArea : int
                maxiaml area of the bounding box of a word
             minconf : float
                minimal confidance of the found word (see the conf output from pytesseract)
             minBlobAreaPix : int
                the minimal number of pixels in a connected component.
                connected component with number of pixels below this number will be removed
             min_v_area : int, default = 30
               minimal number of pixels indicating a valid row of measurements, when looking for the V sign
               indicator from Optopol


            Output
            -------
            data : dictionary
              output dictionary from pytesseract
            res  : DataFrame
              organized extracted values in a DF, with columns
              corresponding to AxialLength (mm), ACD (mm), LT(mm), CCT(mm)
              the rows correspond to 10 measurementsm and the Avg to the average
              (as computed by the OCT)
        """
        imc    = self._GetCombinedBlocksImage(img,eye)
        imc    = self._RemoveBlobs(imc,minArea=minBlobAreaPix)
        # custom_config = r'-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz!@#$%^&*()¢ --psm 6'

        data   = pytesseract.image_to_data(imc,lang='eng',output_type=Output.DICT)#,config="--psm 10 --oem 10 -c tessedit_char_whitelist=0123456789.")
        width  = np.asanyarray(data['width'],dtype=np.int)
        height = np.asanyarray(data['height'],dtype=np.int)
        c      = np.asanyarray(data['conf'],dtype=np.float)
        inds   = (c>minConf)&(width>minWidth)&(height>minHeight)&((width*height)<maxArea)

        # Filter dictionaries
        for kIdx in data.keys():
            data[kIdx]= np.asanyarray(data[kIdx])[inds]
        # organize output
        # re-sort by rows and columns
        data['col_num'] = np.zeros(len(data['line_num']))
        rowNum=0
        if eye.lower()=='right':
            rows = self.rows_r
            cols = self.cols_r
        elif eye.lower()=='left':
            rows = self.rows_l
            cols = self.cols_l

        for rIdx in np.arange(0,len(rows)-1,2):
            data['line_num'][np.where((data['top']>=rows[rIdx])&(data['top']<=rows[rIdx+1]))]=rowNum
            rowNum+=1

        colNum=0
        for cIdx in np.arange(0,len(cols)-1,2):
            data['col_num'][np.where((data['left']>=cols[cIdx])&(data['left']<=cols[cIdx+1]))]=colNum
            colNum+=1

        # arrange values in a table
        res = pd.DataFrame(index =[0,1,2,3,4,5,6,7,8,9,'Avg'],columns=['AxialLength','ACD','LT','CCT','valid'])

        # Check for row validity according to the Optopol V mark located to the left of the row of measurements
        # in the bmp report. Those are invalif measurements which are not  taken into account while computing the mean,
        # and are colored light gray. We want to exclude thwse measurements
        # to detect those measurments, we look for the v mark to the left of the row
        res.loc['Avg','valid'] = True
        if eye.lower()=='right':
            rows = self.rows_r
            cols = self.marks_r
        elif eye.lower() =='left':
            rows = self.rows_l
            cols = self.marks_l
        for rIdx in range(10): # for each line
            v_img = img[int(rows[2*rIdx]):int(rows[2*rIdx+1]),cols[0]:cols[1]]
            if (v_img==0).sum()>min_v_area:
                res.loc[res.index[rIdx],'valid'] = True
            else:
                res.loc[res.index[rIdx],'valid'] = False

        for dIdx in range(len(data['text'])):
            rowInd = res.index[int(data['line_num'][dIdx])]
            colInd = res.columns[int(data['col_num'][dIdx])]
            # clean the output string
            val    = data['text'][dIdx].replace(',','.')
            re_val = re.findall('\d+\.\d+',val)

            # TODO: pass the val extracted to a processing method to eliminate/correct bad strings
            # correct for cases in which the decimal point appears at the end position

            if len(re_val)==1:
                val = re_val[0]

            valid  = self._CheckValueValidity(val)
            if valid:
                if len(re.findall(r'\.',val))==0:
                    if re.findall('cct',colInd):
                        # transform to a number
                        # val = float(val)/1000
                        res.loc[rowInd,colInd] = float(val)/1000
                    else:
                        # val = float(val)/100
                        res.loc[rowInd,colInd] = float(val)/100
                else:
                    # val = float(val)
                    res.loc[rowInd,colInd] = float(val)

        # re-compute averages, if missing, using available data
        res = self._RecomputeAverages(res,eye)

        return data,res

    @staticmethod
    def _RemoveBlobs(imc,minArea=4):
        """
            Remove connected components of area less than minArea
            input image, imc, must be b&W
            output is imc without the isolated regions
        """
        _, _, stats, _ = cv2.connectedComponentsWithStats(255-imc, connectivity=8)
        for sIdx in range(len(stats)):
            if stats[sIdx,4]<minArea:
                row    = stats[sIdx,1]
                col    = stats[sIdx,0]
                width  = stats[sIdx,2]
                height = stats[sIdx,3]
                imc[row:(row+height),col:(col+width)] = 255
        return imc

    def _RecomputeAverages(self,res,eye):
        """
           In case the averages are not well extracted by OCR, use existing valid measurements
           to recumpute the mean value of each row

           Parameters
           -----------
           res : DataFrame
             result dataframe
           eye : str
            left or right eye

            Returns
            ---------
            res : DataFrame
            result data frame with Avg index re-computed
        """
        for kIdx in ['AxialLength','ACD','LT','CCT']:
            if pd.isna(res.loc['Avg',kIdx]):
                print(f'[Warning][RevoParser] The average value of {kIdx} in {eye} eye was not extracted properly.')
                # Detect acceptable rows
                vals = res.loc[res['valid'],kIdx]
                vals.drop('Avg',inplace=True)
                vals = vals.dropna()
                if len(vals)>0:
                    n_digits = 3 if kIdx.lower()=='cct' else 2
                    res.loc['Avg',kIdx] = round(vals.mean(),ndigits=n_digits)
                    print(f"[Info][RevoParser] Re-computed average value for {kIdx} in {eye} eye to {res.loc['Avg',kIdx]}")
                else:
                    print(f'[Warning][RevoParser] Could not find valid measurements in {kIdx} of {eye} eye to re-compute the average.')
                    res.loc['Avg',kIdx] = None

        return res

    def _CheckValueValidity(self,val):
        # examine if val has letters in it
        excVals = re.findall(r'[^0-9\.]',val)
        if len(excVals)==0:
            try:
                float(val)
                return True
            except:
                print(f'[Info][RevoParser] Value {val} did not pass the validity test and will be set to nan')
                return False
        else:
            print(f'[Info][RevoParser] Value {val} did not pass the validity test and will be set to nan')
            return False
