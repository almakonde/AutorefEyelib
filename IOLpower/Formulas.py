
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# TODO: investigate the relationship between ACD and post operative ELP
# TODO: obtain post-op ACD post-op axial length
# TODO: find relationship between the cornea radius or curvature and the axial length
# TODO: integrate Holladay biometry screening criteria for left-right eyes
# TODO: complete BarretteUniversalII

class Formulas:

    def ComputeAllFormulas(self,formulas,Aconst,meanK,acd,wtw,axialLength,Rt,meanCornealHeight,meanACD,surgeonFactor,hofferQPersonalizedACD,pDelta=0.5):
        """
         Compute all formulas, and their predicted biophysical measurements such as ELp and axial length

         Output:
         -------
         P, DataFrame,
          IOL power (D)
         R, FataFrame
          predicted refraction (D)
         E, DataFrame
          predicted ELP (mm)
         A, DataFrame
          predicted axial length (mm)
        """

        P  = pd.DataFrame(columns=formulas)
        R  = pd.DataFrame(columns=formulas)
        E  = pd.DataFrame(columns=formulas)
        A  = pd.DataFrame(columns=formulas)
        for fIdx in formulas:
            if fIdx.lower()=="srkt":
                P['SRKT'],R['SRKT'],E['SRKT'],A['SRKT']             =\
                    self.SRKT(Aconst,meanK,axialLength,Rt,meanCornealHeight=meanCornealHeight,pDelta=pDelta)
            if fIdx.lower()=="t2":
                P['T2'],R['T2'],E['T2'],A['T2']             =\
                    self.SRKT(Aconst,meanK,axialLength,Rt,meanCornealHeight=meanCornealHeight,pDelta=pDelta,T2=True)
            if fIdx.lower()=="shammas":
                P['Shammas'],R['Shammas'],E['Shammas'],A['Shammas'] =\
                     self.Shammas(Aconst,meanK,axialLength,Rt,pDelta=pDelta)
            if fIdx.lower()=="binkhorst-2":
                P['Binkhorst-2'],R['Binkhorst-2'],E['Binkhorst-2'],A['Binkhorst-2']  = \
                    self.Binkhorst2(meanK,axialLength,meanACD,Rt,pDelta=pDelta)
            if fIdx.lower()=="haigis":
                P['Haigis'],R['Haigis'],E['Haigis'], A['Haigis']    = \
                    self.Haigis(Aconst,meanK,acd,axialLength,Rt,personaliseELP=True,pDelta=pDelta)
            if fIdx.lower()=="barrett-1":
                P['Barrett-1'],R['Barrett-1'],E['Barrett-1'],A['Barrett-1'] =\
                    self.BarrettUniversal1(meanK,acd,axialLength,Rt,pDelta=pDelta)
            if fIdx.lower()=="holladay-1":
                P['Holladay-1'],R['Holladay-1'],E['Holladay-1'],A['Holladay-1'],_  = \
                    self.HolladayI(Aconst,meanK,axialLength,Rt,sFactor=surgeonFactor,pDelta=pDelta)
            if fIdx.lower()=="hoffer-q":
                P['Hoffer-Q'],R['Hoffer-Q'],E['Hoffer-Q'], A['Hoffer-Q'] =\
                    self.HofferQ(meanK,axialLength,hofferQPersonalizedACD,Rt,pDelta=pDelta)
            if fIdx.lower()=="olsen":
                P['Olsen'],R['Olsen'],E['Olsen'],A['Olsen']  =\
                     self.Olsen(meanK,acd,wtw,axialLength,meanACD,Rt,pDelta=pDelta)
        return P,R,E,A

    def GetParams(self):
        params = {"SRKT":{"n_c":1.333,"n_v":1.336,},
                    "T2":{"n_c":1.33,"n_v":1.336},
                    "Barrett-1":{"n_c":1.336,"n_v":1.336},
                    "Hoffer-Q":{"n_c":1.336,"n_v":1.336},
                    "Olsen":{"n_c":1.3315,"n_v":1.336},
                    "Shammas":{"n_c":1.333,"n_v":1.336},
                    "Haigis":{"n_c":1.332,"n_v":1.336},
                    "Holladay-1":{"n_c":1.336,"n_v":1.336},
                    "Binkhorst-2":{"n_c":1.336,"n_v":1.333}
                  }
        return params
    @staticmethod
    def MeanCornealHeight(wtw,r,al):
        """
            Compute the cornea height using retrospecive data to be used with the SRKT formula

            Parameters:
            ----------
             wtw, float
              white to white (mm)
             r, float
              mean corneal radius (mm)
             al, float
              axial length (mm)

              Output:
              --------
              mean corneal height, float
        """

        inds = np.where((np.isnan(r)==False)&(r>0)&\
                        (np.isnan(wtw)==False)&(wtw>0)&\
                        (np.isnan(al)==False)&(al>0))[0]
        r    = r.iloc[inds].values
        wtw  = wtw.iloc[inds].values
        al   =  al.iloc[inds].values
        k    = 337.5/r # mean corneal power (D)
        # Adjust the wtw according to the axial length and keratometry
        wtw = np.where(al<=24.2,0.098*k+0.58412*al-5.41,\
                                0.098*k+0.58412*(-0.0237*al**2+1.716*al-3.446)-5.41)
        desc = r**2 -(wtw/2)**2
        inds = np.where(desc>=0)[0]
        # compute corneal height
        h    = r[inds]-(desc[inds])**0.5

        return np.mean(h) # return average (mm)

    @staticmethod
    def WtwSRKT(al,k):
        """
            Compute the white to white (wtw) using the axial length (mm) and keratometry (D)
            to be used in the SRK/T formula

            Parameters:
            -----------
            al - array(float)
                axial length (mm)
            k  - array(float)
                mean keratometry (D)
            Returns
            -------
            wtw - array(float)
                white to white corneal diameter (mm)
        """

        wtw = np.where(al<=24.2,0.098*k+0.58412*al-5.41,\
                        0.098*k+0.58412*(-0.0237*al**2+1.716*al-3.446)-5.41)
        return wtw

    @staticmethod
    def ComputeAconst(meanK,al,Pi):
        """
            Compute the A-constant by multi-linear regression using post-op data.

            Parameters:
            -----------
            meanK, DataFrame
             mean keratometery (D)
            al, DataFrame
             axial length (mm)
            Pi, DataFrame
              implanted IOL power (D)

            Output:
            -------
            Aconst, float
             the database specific A-constant
        """

        inds = np.where(al.notna())&(al>0)&\
                        (meanK.notna())&(meanK>0)&\
                        (Pi.notna())[0]
        al = al.iloc[inds].values
        k  = meanK.iloc[inds].values
        p  = Pi.iloc[inds].values
        X  = pd.DataFrame()
        X["axialLength"] = al
        X["k"] = k
        Y = pd.DataFrame(p)
        l = LinearRegression(fit_intercept=True)
        l.fit(X,Y)
        return l

    @staticmethod
    def PersonalizeAConst(Aconst,meanK,axialLength,Rf):
        """
            compute the A-constant based on the dataset and the SRK formula
            Parameters:
            -----------
            Aconst- array(float)
                A-constant from manufacture
            axialLength- array(float)
                axial length (mm)
            Rr- array(float)
                observed refraction post surgery
            meanK- array(float)
                average keratometry (D)
        """

        # Truncate invalid entries
        inds         = np.where((axialLength.notna())&(axialLength>0)&\
                                (meanK.notna())&(meanK>0)&\
                                (Rf.notna()))[0]
        axialLength = axialLength.iloc[inds].values
        meanK       = meanK.iloc[inds].values
        Rf          = Rf.iloc[inds].values

        alpha = np.zeros(shape= (len(axialLength),))
        beta  = np.zeros(shape= (len(axialLength),))
        for lIdx in range(len(alpha)):
            if axialLength[lIdx]>=24:
                alpha[lIdx] = -0.5
            elif axialLength[lIdx]>=22 and axialLength[lIdx]<24:
                alpha[lIdx] = 0
            elif axialLength[lIdx]>=21 and axialLength[lIdx]<22:
                alpha[lIdx] = 1
            elif axialLength[lIdx]>=20 and axialLength[lIdx]<21:
                alpha[lIdx] = 2
            elif axialLength[lIdx]<20:
                alpha[lIdx] = 3
            # compute beta
            P = Aconst-2.5*axialLength[lIdx] -0.9*meanK[lIdx]
            beta[lIdx] = 1.25 if P>14  else 1

        # compute the new personalized A constant
        Ap = np.mean(Aconst-alpha-beta*Rf)
        return Ap

    @staticmethod
    def HoffereMeanELP(meanK,al,Rf,Pi,v=0.012):
        """
            Compute the mean ELP according to Hoffer using post-op data
            Parameters:
            ----------
            al - array(float)
                axial length mm
            meanK- array(float)
                average keratometry
            Rf - array(float)
                final refraction (D)
            Pi - array(float)
                implanted IOL power (D)
            v- float
                vertex distance (m)
            Output:
            -------
            average individualized elp (mean_elp)
        """
        Rrm  = Rf/(1-v*Rf) # translate to the corneal plane
        d    = np.zeros(shape=(len(al,)))
        g    = 1.336/(meanK+Rrm)
        desc = g**2 +4*(1.336*(g-al))/Pi
        inds = np.where(desc>=0)[0]
        d    = 0.5*(al[inds]+g[inds]-(desc[inds])**0.5)-5e-5

        return np.mean(d)

    def BarrettUniversal1(self,meanK,acd,al,Rt,v=0.012,pDelta=0.5):
        """
         Compute IOL power by the Barrett universal 1 formula.
         Based on:
         Barrett, Graham D.
         "Intraocular lens calculation formulas for new intraocular lens implants."
         Journal of Cataract & Refractive Surgery 13.4 (1987): 389-396.

         Parameters:
         -----------
         al- float
             axial length (mm)
         meanK, float
            average keratometry D
         acd, float
            anterior chamber depth (mm)

         Output:
         ------
          P- float,
             predicted IOL implant power (D)
          R- float
             predicted refraction at corneal plane (D)
          elp- float
             predicted effective lens position (mm)
          A - float
             predicted axial length (mm). same as input
        """

        # Dld - designated power of IOL
        # FL1 - power of the anterior surface of implant
        # FL2 - power of the posterior surface of the implant
        # T   - thicknes of the implant
        # N2  - refractive index of the lens implant
        # N1  - refractive index of aqueous
        Ri  = 25    # radius of the posterior IOL # todo: use gullstrand approximation
        N2  = 1.435 # refractive index lens
        N1  = 1.336 # refractive index cornea
        F2  = (N2-N1)/Ri  # m
        P1  = 21.5   # D radius of curvature
        D   = acd/1000# default=0.0048 # m
        T   = 0.001  # m lense thickness
        C   = meanK + Rt/(1-v*Rt) # D
        A   = al/1000 # m
        for _ in range(40):
            F1  = (P1-F2)/(1-T*F2/N2) # D power of anterior IOL
            E2  = (N1/N2)*F1*T/P1 # m  distance to second principle plane
            E1  = (N1/N2)*F2*T/P1 # m
            L   = (N1/C)
            U   = L-D-E1 # m
            V   = A-D-T+E2 # m
            P2  = (N1/V)-(N1/U)
            P1 = P2
        if pDelta is not None:
            P   = round(P2/pDelta)*pDelta # power of IOL
        else:
            P = P2
        elp = (D +(N1/N2)*F2*T/P)*1000 # the effective lens position (mm)
        R   = self.PredictedRefraction(P,meanK,al,elp,n_c=N1,n_v=N1)
        R   = R/(1+v*R) # translate back to corneal plane
        return P,R,elp,al

    def BarrettUniversal2(self,meanK,acd,al,Rt,Aconst):
        """
         NOTE: UNFINISHED
         Predict IOL power using the Barrett Universal 2 IOL formula
         Based on:
         Barrett, Graham D.
         "An improved universal theoretical formula for intraocular lens power prediction."
         Journal of Cataract & Refractive Surgery 19.6 (1993): 713-720.

         Parameters:
         -------
         al - float
            axial length [m]
         K -
            keratometry
         Pz - P factor of the cornea
        """
        # First, compute the lens factor, which gives the location of the second principle plane from the iris plane
        LF  = Aconst*0.5825-67.6627 # for average eye of axial length=23.5mm and keratometry=43.8D
        # then compute the lens thickness
        RA = (N2-N1)*1000/(P1/2)
        T = RA - np.sqrt(Ra**2 -(OD/2)**2) + (RP-np.sqrt(RP**2-(OD/2)**2))

        RG = 0.35066*al-0.06607*meanK+5.70871




        #==========
        Rc  = self.K2Rc(meanK) # central anterior radius of the cornea
        Rcc = Rc*0.883 #[m] central posterior radius of the cornea (using Gullstrand ratio)
        Kc  = 376/Rc - (40/Rcc) +(0.00052/1.376)*(376/Rc)*(40/Rcc) # power of the cornea
        RG  = 0.35066*al-0.0667*Kc+5.70871 # radius of the globe (posterior)
        PZ  = None # should be adjusted until the radius of peripheral cornea is between 11.5 and 13.5
        Pz  = 0.1
        for _ in range(10):
            RCP = (np.sqrt(Rc**2+(1-PZ)*25)**3)/Rc**2 # radius of the peripheral cornea
            # ACD = al -0.593 +0.13 -RG - np.sqrt(RG**2 -RCP**2 +np.sqrt(RCP-ACD)) # solve for the ACD
            # rearranging and solving for the ACD
            # elp = (RG**2 -(al-0.593+0.13)**2)/(2*RCP-2*(al-0.593+0.13))
            elp = al -0.593+0.13-RG -np.sqrt(RG**2-RCP**2 +(RCP-acd)**2)

        P = self.ThinLensPower(elp,Kc,al,Rt)
        return P

    def SRK(self,Aconst,meanK,Rt,axialLength=None):
        """
         Compute the IOL powere using the SRK I formula

         Parameters:
         -----------
         Aconst, float,
             lens A constant
         K, float
            keratometry in diopter using index 1.3375
         axialLength, float, default=Nnone
            axial length (m)
            if unspecified or is None, theal will be estimated from the data
         Rt - float,
            target refraction (D)
        """

        if axialLength==None: # if axial length is not available, estimate it using the mean K
            R_c = self.K2Rc(meanK) # in mm
            # try estimateing the axial length post -op
            axialLength  = self.Rc2Al(R_c) # in mm

        Pe = Aconst-0.9*meanK-2.5*axialLength # IOL power predicted
        P  = Pe - 1.5*Rt # IOL power implanted to obtain Rt
        return P

    def SRK2(self,Aconst,meanK,axialLength,Rt,n_c=1.336):
        """
            Compute the second-generation SRK II formula

            Parameters:
            --------
            Aconst, float
                the A constant
            meanK, float
                mean keratometry in diopters (K1+K2)/2
            axialLength, float, positive, default =None
                axial length in mm.
                when al=None or unspecified, it is estimated using the mean keratometry value
            Rt- float,
                target refraction (D)

            Output:
            -------
             IOL power (D), float
        """
        # Estimate the axial length post-op if not specified using keratometry
        axialLength = np.where(np.isnan(axialLength),self.K2Al(meanK),axialLength)
        # AconstOrig  = Aconst.copy()# self.PersonaliseAConst(Aconst)
        P     = np.zeros(len(axialLength))
        alpha = np.zeros(len(axialLength))
        for lIdx in range(len(axialLength)):
            # Aconst = AconstOrig.copy()
            # correction to the Aconst
            if axialLength[lIdx]>=24.5:
                alpha[lIdx]=-0.5
            elif axialLength[lIdx]>=21 and axialLength[lIdx]<22:
                alpha[lIdx]=1
            elif axialLength[lIdx]>=20 and axialLength[lIdx]<21:
                alpha[lIdx]=2
            elif axialLength[lIdx]<20:
                alpha[lIdx]=3

            # compute the required IOL power to obtain Rt
            P[lIdx] = Aconst-0.9*meanK[lIdx]-2.5*axialLength[lIdx]+alpha[lIdx]

        P = np.where(P>14,P-1.25*Rt,P-Rt)
        return P

    def SRKT(self,Aconst,meanK,axialLength,Rt,meanCornealHeight=3.336,v=0.012,n_a=1.336,n_c=1.333,pDelta=0.5,T2=False):
        """
            Compute IOL power by SRK/T formula.

            Reference:
            Retzlaff, John A., Donald R. Sanders, and Manus C. Kraff.
            "Development of the SRK/T intraocular lens implant power calculation formula."
            Journal of Cataract & Refractive Surgery 16.3 (1990): 333-340.

            Parameters:
            -----------
            Aconst- float,
                A-constant for the IOL used (D).
                A personalized A-constant should be used
                see self.PersonalizeAConst procedure
            axialLength- float,
                the axial length (mm)
            meanK - float
                average keratometry (D)
            Rt- float,
                target refraction (D)
            v- float, optional, default=12
                vertex distance between spectacle plane and corneal place
                in mm units
            pDelta- float, optional, default=0.5
              dioptric intervals of the IOL power
            T2- boolean, optional, default=False
                use the T2 algorithm to estimate the corneal height using regression formula
                using T2=True, implements the algorithm of Richard M. Sheard 2010
            n_a- float, optional, default=1.336
                refractive index acquous medium
            n_c- float, optional, default = 1.333
                refractive index cornea

        """
        # meanCornealHeight = self.MeanCornealHeight() # default SRKT=3.336 mm
        P    = np.zeros(axialLength.shape)
        R    = np.zeros(axialLength.shape) # expeted refraction
        elp  = np.zeros(axialLength.shape) # effective len position
        lOpt = np.zeros(axialLength.shape) # optical axial length

        if isinstance(axialLength, (pd.DataFrame,pd.Series)):
            axialLength = axialLength.values
        if isinstance(meanK, (pd.DataFrame,pd.Series)):
            meanK = meanK.values
        if isinstance(Rt,(pd.DataFrame,pd.Series)):
            Rt = Rt.values


        # First, adjust the A-constant for each measurement
        for lIdx in range(len(axialLength)):
            if axialLength[lIdx]>=24:
                alpha = -0.5
            elif axialLength[lIdx]>=22 and axialLength[lIdx]<24:
                alpha = 0
            elif axialLength[lIdx]>=21 and axialLength[lIdx]<22:
                alpha = 1
            elif axialLength[lIdx]>=20 and axialLength[lIdx]<21:
                alpha = 2
            elif axialLength[lIdx]<20:
                alpha = 3

            # correction for long eyes
            if axialLength[lIdx]>24.2:
                # adjust the axial length
                # lCor = -3.36+1.716*axialLength[lIdx]-0.0237*axialLength[lIdx]**2
                lCor = -3.446+1.716*axialLength[lIdx]-0.0237*axialLength[lIdx]**2
            else:
                lCor = axialLength[lIdx]

            # compute corneal width
            cw = -5.4098+0.58412*lCor+0.098*meanK[lIdx]
            # compute the elp as the sum of corneal height and and offset, i.e. elp= h+o, in units of mm,
            # with h the corneal height and o the offset term
            rc  = 337.5/meanK[lIdx] # mean cornea radius of curvature (mm)
            # Compute the corneal height
            desc  = (rc**2 -(cw/2)**2)
            if desc>=0:
                ch   = rc-(rc**2 -(cw/2)**2)**0.5
            else:
                ch = rc

            if T2: # implement the T@ algorithm for the corneal height
                ch = -11.980 + 0.38626*lCor +0.14177*meanK[lIdx]

            # # Use the personalized A-const. To compute the personalized ACD-const
            ACD_const = self.Aconst2AcdConst(Aconst+alpha)# individualized ACD-const
            offset    = ACD_const - meanCornealHeight # personalized offset value (mm)
            # # Compute the elp
            elp[lIdx]  = (ch +offset) # translate to m units
            # compute retinal thickness
            rt       = 0.65696-0.02029*axialLength[lIdx]
            # compute the optical axial length
            lOpt[lIdx] = axialLength[lIdx]+rt

            # Compute IOL power
            Ap        = 1000*n_a*(n_a*rc-(n_c-1)*lOpt[lIdx]-0.001*Rt[lIdx]*(v*(n_a*rc-(n_c-1)*lOpt[lIdx])+lOpt[lIdx]*rc))
            Bp        = (lOpt[lIdx]-elp[lIdx])*(n_a*rc-(n_c-1)*elp[lIdx]-0.001*Rt[lIdx]*(v*(n_a*rc-(n_a-1)*elp[lIdx])+elp[lIdx]*rc))
            if pDelta is not None:
                P[lIdx]  = round((Ap/Bp)/pDelta)*pDelta
            else:
                P[lIdx] = (Ap/Bp)
            # Compute expected refraction at corneal plane
            Ar       = 1000*n_a*(n_a*rc-(n_c-1)*lOpt[lIdx])-P[lIdx]*(lOpt[lIdx]-elp[lIdx])*(n_a*rc-(n_c-1)*elp[lIdx])
            Br       = n_a*(v*(n_a*rc-(n_a-1)*lOpt[lIdx])+lOpt[lIdx]*rc)-0.001*P[lIdx]*(lOpt[lIdx]-elp[lIdx])*(v*(n_a*rc-(n_c-1)*elp[lIdx])+elp[lIdx]*rc)
            R[lIdx]  = Ar/Br

        return P,R, elp, lOpt

    def ElpSrkt(self,Aconst,meanK,axialLength,meanCornealHeight):
        alpha = 0
        if axialLength>=24:
                alpha = -0.5
        elif axialLength>=22 and axialLength<24:
            alpha = 0
        elif axialLength>=21 and axialLength<22:
            alpha = 1
        elif axialLength>=20 and axialLength<21:
            alpha = 2
        elif axialLength<20:
            alpha = 3

        lCor = np.where(axialLength<24.2,axialLength, -3.446+1.715*axialLength-0.0237*axialLength**2)

        cw   = -5.41 +0.58412*lCor+0.098*meanK
        # compute the elp as the sum of corneal height and and offset, i.e. elp= h+o, in units of mm,
        #  with h the corneal height and o the offset term
        rc  = 337.5/meanK # mean cornea radius of curvature (mm)
        # Compute the corneal height
        # print(f'ax:{axialLength},axCor:{lCor}, cw:{cw}, det: {rc**2 -(cw/2)**2}')
        ch   = rc-(rc**2 -(cw/2)**2)**0.5
        # Use the personalized A-const. To compute the personalized ACD-const
        ACD_const = self.Aconst2AcdConst(Aconst+alpha) # individualized ACD-const
        offset    = ACD_const - meanCornealHeight      # personalized offset value (mm)
        # Compute the elp
        elp      = (ch +offset) # effective lens position (mm)
        return elp

    @staticmethod
    def Aconst2AcdConst(Aconst): #V
        """ Translate the A constant (float) to ACD constant"""
        return 0.62467*Aconst-68.747 # default = 4 mm

    @staticmethod
    def Aconst2SurgeonFactor(Aconst):
        return 0.5663*Aconst-65.6

    @staticmethod
    def ThinLensPower(elp,k,al,Rt,n_c=1.336,n_v=1.337,v=0.012,pDelta=0.5):
        """
            Compute IOL power using the thin lens approximation.
            The power is given in the spectacle plane

            Parameters:
            -----------
            n_c, float, default=1.336
                refractive index of the anterior chamber
            n_v, float, default=1.337
                refractive index in the posterior segment
            elp, float
                effective lens position (mm)
            K, float
                corneal power (D)
            al, float
                axial length (mm)
            Rx, float
                target refraction (D)
            v, float
                vertex distance (m)
            pDelta, float, default =0.5
             rounding of the IOL power to the nearest pDelta diopter
             set to None for no rounding
            Output:
            --------
            IOL power (D)
        """

        P = n_v/((al-elp)/1000) - n_c/((n_c/(k+Rt)) - elp/1000)
        if pDelta is not None:
            return np.round(P/pDelta)*pDelta
        else:
            return P

    def Colenbrander(self,elp,k,al,Rt,n_a=1.336,n_v=1.3337,v=0.012):
        """
            Compute IOL power by Colebrander's formula
            all parameters in units of m

            Parameters:
            ----------
             elp, float,
                 effective lens position (m)
             k, float,
                 mean keratometry (D)
             al, float
                 axial length (m)
             Rt, float,
                 target refraction (D)
             n_a, float, default = 1.336
                 refractive index aqueous
             n_v, float, default = 1.337
                 refractive index vitreous
             v, float, default =0.012
                 vertex distance, positive
            Output:
            -------
             P, float,
                predicted IOL power (D)
             R, float
                predicted refraction for iol power P
        """
        elp = elp+5e-2
        P   = self.ThinLensPower(elp,k,al,Rt/(1+v*Rt),n_a,n_v)
        R   = self.PredictedRefraction(P,k,al,elp,n_c=n_a,n_v=n_v)
        R   = R/(1-v*R) # translate to corneal plane
        return P,R,elp,al

    def Fyodorov(self,elp,k,al,R_t,n_a=1.336,n_v=1.336):
        """
            Compute IOL power by Fyodorov's formula.
            Fyodorov's formula is equivalent to the thin lens formula when n_a=n_v.
            In the derivation of the Fyodorov, the ratio nv/na is assumed 1
            Parameters:
            ----------
            elp, float
                effective lens position (m)
            k, float
                mean keratometry (D)
            al, float
                axial length (m)
            R_t, float
                target refraction (D)
            Output:
            ----------
            P, float
                predicted IOL power
        """
        P = (n_v-al*(k+R_t))/((al-elp)*(1-(k+R_t)*elp/n_a ))
        R = self.PredictedRefraction(P,k,al,elp,n_v=n_v,n_c=n_a)
        return P,R,elp, al

    def VanDerHeijde(self,elp,k,al,Rt,n_c=1.336):
        """
          Compute the IOL power by the Van Der Heijde's formula. equivalent to the thin lens formula
        """
        P = n_c/(al-elp) - 1/(1/(k+Rt)-elp/n_c)
        R = self.PredictedRefraction(P,k,al,elp,n_c = n_c)
        return P,R

    @staticmethod
    def CorneaPower(r,n):
        """
            Compute the cornea power

            Parameters:
            --------
            r, float
                radius of the cornea (mm)
            n, float
                refractive index of the cornea
            output:
            ------
            cornea power (D)
        """
        return (n-1)/r

    @staticmethod
    def ThickLensPower(nc,nv,Ra,Rp,ccp):
        """
         Compute the power of a lens using thick lens formula
         Parameters:
         -----------
         nc- float
             refractive index of the anterior surface (usually cornea)
         nv- float
             refractive index of the posterior surface (usually vitreous)
         Ra - float
             radius of the anterior surface (anterior cornea)
         Rp - float
             radius of the posterior surface (posterior cornea)
        """
        Ka = (nc-1)/Ra
        Kp = (nv-nc)/Rp
        P  = Ka +Kp(1-ccp*Ka/nc)
        return P

    @staticmethod
    def HaigisELP(meanK,acd,al,Pi,Rf,na=1.3315,v=0.012):
        """
            Mean ELP for the Haigis formula

            Parameters:
            ----------
            meanK, DataFrame
              mean keratometry (D): (k1+k2)/2
            acd, DataFrame
              anterior chamber depth (mm)
            al, DataFrame
             axial length (mm)
            Pi, DataFrame
             IOL power implanted (D)
            RF, DataFrame
             refraction post op (D)
            n, float, optional, default=1.3315
            v, float, optional, default=0.012
             vertex distance (m)

            Output:
            -------
            elp, float
              effective lens position (mm)
        """

        aInds   = np.where((acd.notna())\
                           & (acd>0)\
                           & (al.notna())\
                           & (al>0)\
                           & (al<28)\
                           & (Pi.notna())\
                           & (Pi!=0)\
                           & (Rf.notna())\
                           & (meanK.notna())\
                           & (meanK>0))[0]
        acd   = acd.iloc[aInds].values
        al    = al.iloc[aInds].values
        Pi    = Pi.iloc[aInds].values
        Rf    = Rf.iloc[aInds].values
        meanK = meanK.iloc[aInds].values
        # meanACD = np.mean(ACD.iloc[aInds & lInds])
        # meanAL  = np.mean(AL.iloc[aInds & lInds])

        # set default values
        # a1 = 0.4
        # a2 = 0.1
        # a0 = self.Aconst2AcdConst(Aconst)# 0.62467*Aconst-72.434 # ACD_const -a1*meanACD-a2*meanAL

        # Rrm      = R/(1-v*R) # translate from spectacle to cornea plane
        # nu       = n/(Rrm+k)
        # d        = (pi*(al+nu) -np.sqrt((pi*(al+nu))**2 - 4*pi*(n*(al-nu)+pi*al*nu)))/(2*pi)
        # l        = LinearRegression(fit_intercept=True)
        # X        = pd.DataFrame()
        # Y        = pd.DataFrame()
        # X["acd"] = acd-np.mean(acd)
        # X["al"]  = al-np.mean(al)
        # Y["y"]   = d
        # l.fit(X,Y)
        # a0      = l.intercept_[0]
        # a1      = l.coef_[0][0]
        # a2      = l.coef_[0][1]
        #####################33
        # n = 1.336
        # z       = meanK+ Rf/(1-v*Rf) # spectacle plane
        L       = al/1000
        # elp  = 0.5*(L+na/z)-(0.25*(L+na/z)**2-(L*na/z-(na**2)/(Pi*z)+na*L/Pi))**0.5
        nu      = na/(meanK+Rf) # m units
        desc    = ((L-nu)**2 +4*na*(nu-L)/Pi) # m^2
        inds    = np.where(desc>=0)[0]
        elp     = ((L[inds]+nu[inds]) - desc[inds]**0.5)/2
        meanELP = 0.001*elp[np.isnan(elp)==False].mean()

        # perform linear regression to retrieve the coefficients a0,a1,a2 according to d=a0+a1*AL+a2*ACD
        lr       = LinearRegression(fit_intercept=True)
        X        = pd.DataFrame()
        Y        = pd.DataFrame()
        X["acd"] = acd[inds]#-np.mean(acd[inds])
        X["al"]  = al[inds] #-np.mean(al[inds])
        Y["y"]   = elp*1000
        lr.fit(X,Y)
        a0      = lr.intercept_[0]
        a1      = lr.coef_[0][0]
        a2      = lr.coef_[0][1]
        return meanELP, a0,a1,a2

    def Haigis(self,Aconst,meanK,acd,axialLength,Rt,v=0.012,n_v=1.336,n_c=1.332,personaliseELP=True,pDelta=0.5):
        """
            Compute the IOL power according to Haigis method.


            Parameters:
            ------------
            axialLength - float, default = None
                axial length [m]
                if the axial length is not specified or is None,
                it is estimated by the linear regression using mean curvature
                (Zheng, 2005)
            acd - float
                anterior chamber depth [m]
            meanK- float
                the average keratometry (K1+K2)/2 in units of (D)
            wtw - float, default=None
                white to white, twice the cornea radius [m] ( or white to white)
                if R_c=wtw/2 is not specified or it is None,
                then it is estimated by 337.5/meanK
            Rt - float
                target refraction (D)
            Aconst - float
                manufacturer A-constant for the IOL
            v   - float, default=12 mm (0.012m)
                vertex distance (m)
            n_v - float, default=1.336
                refractive index vitrous
            n_c - float, default=1.3315
                refractive index cornea

            Output:
            ------------
            P - float,
                IOL power (D)
            R, float
               predicted refraction from (rounded) predicted power
            elp, float
             predictred effective lens position (mm)
            axialLength, float
             predicted axial length (mm)
        """


        if isinstance(meanK,pd.Series):
            meanK = meanK.values
        if isinstance(acd, pd.Series):
            acd = acd.values
        if isinstance(axialLength, pd.Series):
            axialLength = axialLength.values
        if isinstance(Rt,pd.Series):
            Rt = Rt.values

        # compute radius of cornea if not supplied
        # Rc = np.where((np.isnan(Rc)|(Rc==0)),self.K2Rc(meanK),Rc)
        # if no axial length is supplied, compute it using correlation with the cornea radius
        axialLength = np.where(np.isnan(axialLength),self.K2Al(meanK),axialLength)

        # Compute the ELP (mm) - default values
        a1    = 0.4 # default value
        a2    = 0.1 # default value
        # a0    = self.Aconst2AcdConst(Aconst)#
        a0    = 0.62467*Aconst -72.434
        # if personaliseELP:
        #     a0,a1,a2,d = self.HaigisELPconst()
        elp   = a0 + a1*acd +a2*axialLength # in mm
        # elp = self.HaigisELP()
        # compute the IOL power at spectacle plane
        # TODO: check if refraction should be computed in the spectacle plane or the corneal
        P = self.ThinLensPower(elp,meanK,axialLength,Rt/(1+v*Rt),n_c=n_c,n_v=n_v,pDelta=pDelta)
        R = self.PredictedRefraction(P,meanK,axialLength,elp,v=v,n_v=n_v,n_c=n_c)


        return P, R, elp, axialLength

    def HaigisL(self,Aconst,meanK,acd,axialLength,Rt,v=0.012,n_v=1.336,n_c=1.332,pDelta=0.5):
        """
         Compute the IOL power based on the Haigis-L formula.
         The Haigis L formula adjusts the IOL master measurements using linear regression formula
         and hen use the adjusted value in the Haigis formula to obtain IOL power

         Parameters:
         -----------

         Reference:
         ---------
         Haigis 2008. Intraocular lens calculation after refractive surgery
         for myopia: Haigis-L formula

        """

         # correct the mean corneal radius
        rc = -5.1625*(337.5/meanK) + 82.2603- 0.35
        meanK = 331.5/rc # convert back to mean keratometry
        # substitute in the Haigis formula to obtain the power

        P, R, elp, axialLength = self.Haigis(Aconst,meanK,acd,axialLength,Rt,v=v,n_v=n_v,n_c=n_c,pDelta=pDelta)
        return P, R, elp, axialLength

    def Olsen(self,meanK,acd,wtw,axialLength,meanACD,Rt,n_v=1.336,n_c=1.3315,v=0.012,pDelta=0.5,rDelta =0.25)->tuple:
        """
         Compute IOL power using the Olsen formula.
         Based on:
         * Olsen T. et al . 1990, Theoretical versus SRK I and SRK II calculation of intraocular lens power
         and,
         * Olsen T et al. 1987. Theoretical approach to intraocular lens calculation using Gaussian optics
         * Olsen 1995. Intraocular lens power calculation with an improved anterior chamber depth prediction algorithm

         Parameters:
         -----------
          acd, float,
            anterior chamber depth pre op. (mm)
          meanK, float
            average keratometry (D)
          axialLength, float
            axial length pre-op (mm)
          wtw, float
            white to white (mm)
          Rc, float
            cornea radius (mm)
          Rt, float
            target refraction (D)

         Returns:
         --------
         P, float
          IOL power (D)
         R, float
          expected refraction based on P, rounded to nearest pDelta diopter
         elp, float
          expetcted lens position (mm)
         axialLength, float
          adjusted axial length (mm)
        """

        # check input data types and lengths
        inputLength = np.zeros(5,dtype=int)
        if isinstance(meanK,pd.Series):
            meanK = meanK.values
            inputLength[0] = len(meanK)
        if isinstance(acd, pd.Series):
            acd = acd.values
            inputLength[1] = len(acd)
        if isinstance(wtw,pd.Series):
            wtw = wtw.values
            inputLength[2] = len(wtw)
        if isinstance(axialLength,pd.Series):
            axialLength    = axialLength.values
            inputLength[3] = len(axialLength)
        if isinstance(Rt,pd.Series):
            Rt = Rt.values
            inputLength[4] = len(Rt)

        if not isinstance(n_c,(float,int)):
            raise ValueError(f'n_c must be numeric but recieved class: {n_c.__class__}')
        elif n_c<=0:
            raise ValueError('n_c must be positive')
        if not isinstance(n_v,(float,int)):
            raise ValueError(f'n_v must be numeric but recieved class: {n_v.__class__}')
        elif n_v<=0:
            raise ValueError('n_v must be positive')
        if not isinstance(v,(float,int)):
            raise ValueError(f'v must be numeric but recieved class: {v.__class__}')
        if not isinstance(pDelta,(float,int,type(None))):
            raise ValueError(f'pDelta must be numeric but recieved class: {pDelta.__class__}')
        if pDelta is not None:
            if pDelta<=0:
                raise ValueError(f'pDelta must be positive, got {pDelta}')
        if not isinstance(rDelta,(float,int,type(None))):
            raise ValueError(f'rDelta must be numeric or None, but recieved class: {rDelta.__class__}')
        if rDelta is not None:
            if rDelta<=0:
                raise ValueError(f'rDelta must be positive, got {rDelta}')
        if not (inputLength==inputLength[0]).all():
            raise ValueError(f'Input arrays must have the same length. Got lengths: {inputLength}')

        # ultrasound constants
        ve = 1550 # effective velocity between cornea and retina m/s
        vl = 1641 # lenticular velocity of sound in the lens m/s
        va = 1532 # aqueous velocity of sound in the anterior chamber m/s
        axialLength = np.where(np.isnan(axialLength),self.K2Al(meanK),axialLength)
        # Rc = 337.5/meanK
        Rc = 1000*(n_c-1)/meanK
        # First, estimate the natiral crystalline lens thickness by axial length in mm
        tl = -0.082*axialLength+6.44 # mm. default= assume a constant 4.5 mm


        # Second, compute the modified axial length
        axialLength = (axialLength/ve - tl/vl)*va + tl

        # Third, compute the mean post-op acd in the db
        # meanACD = self.data['l_acd_mean'].append(self.data['r_acd_mean']).dropna().mean()

        #Fourth, estimate the corneal height
        cornealHeight = Rc-(Rc**2-(wtw/2)**2)**0.5

        # Fifth, compute the ELP
        # elp = (meanACD + 0.12*cornealHeight + 0.33*acd+ 0.3*tl + 0.1*axialLength - 5.18)
        elp = meanACD +0.5*acd +0.1*axialLength+0.15*cornealHeight+0.2*tl-5.38 # olsen 1995
        # from Olsen 1987
        # elp  = 1.97 + 0.26*cornealHeight + 0.28*acd + 0.14*tl
        # elp = 0.37 + 0.26*cornealHeight + 0.28*acd + 0.14*3.9
        # from Olsen 1991
        # elp = meanACD - 3.62 + 0.25*acd + 0.12*axialLength
        # from Olsen 1995 ...

        # Finally, compute the IOL power
        P  = self.ThinLensPower(elp,meanK,axialLength,Rt,n_c=n_c,n_v=n_v,pDelta=pDelta)
        Rs = self.PredictedRefraction(P,meanK,axialLength,elp,v= v,n_c=n_c,n_v=n_v,rDelta=rDelta)
        return P,Rs,elp, axialLength

    def NaeserPosteriorLensPosition(self,age,acd,al):
        """
         Return the position of the posterior lens capsule

        """
        poteriorLensCapsule = 2.4+0.11*age+0.171*acd+0.051*al
        return poteriorLensCapsule

    def Binkhorst2(self,meanK,axialLength,acd_mean,Rt,n_c=1.333,n_v=1.333,v=0.012,pDelta=0.5,rDelta = None ):
        """
            Parameters:
            -------------
            Rt- float,
                postoperative refraction
            v - float, default =12mm
                vertex distance in mm
            n_c- float, default=1.333
                refractive index cornea
            n_v - float, default =1.336
                refractive inde of vitreous
            r - float,
              corneal power in mm radius
            meanK- float,
                average keratometry (K1+K2)/2 in D
        """
        # make sure iput are np.array
        if isinstance(meanK,pd.Series):
            meanK = meanK.values
        if isinstance(axialLength, pd.Series):
            axialLength = axialLength.values
        if isinstance(Rt, pd.Series):
            Rt = Rt.values

        axialLength = np.where(axialLength>26,26,axialLength)
        elp = acd_mean*axialLength/23.45 # v

        P = self.ThinLensPower(elp,meanK,axialLength,Rt,n_c=n_c,n_v=n_v,pDelta=pDelta)
        R = self.PredictedRefraction(P,meanK,axialLength,elp,v=v,n_c=n_c,n_v=n_v,rDelta=rDelta)

        return P,R,elp,axialLength

    def K2Al(self,k):
        """
            Mean keratometry to axial length (mm).
            Based on linear regression with RANSAC on the CIV database
            Parameters:
            -----------
            k, float,
            non negative float, representing the keratometry in D
            Returns:
            --------
            axialLength, float
             predicted axial length (mm) based on the keratometery values inserted
        """
        return 39.139 -0.36*k
        return self.Rc2Al(self.K2Rc(k))

    @staticmethod
    def K2Rc(K,n_v=1.3375):
        """
            Translate keratometry K  (Diopters) to radius of curvature in m
            Parameters:
            ---------
            K, float,
            non negative float repres4nting the keratometry of the cornea in Diopters
            n_v, float, default = 1.3375
             refractive index of vitreous

            Returns:
            --------
            Rc, float
             radius of the cornea (mm)
        """
        return (n_v-1)/K

    @staticmethod
    def Rc2Al(R_c):
        """
            Compute an approximation to the axial length using the radius of cornea
            By using Zheng 2005 linear relationship found by regression
            Parameters:
            ---------
            R_c - radius in m
            Output:
            ---------
            al - axial length in m

        """
        # return 33.87+0.226*R_c
        return  20.5 +0.25*R_c

    def Hoffer(self,Aconst,meanK,axialLength,Rt,v=0.012,pDelta=0.5):
        """
            Compute IOL power by Hoffer formula.
            A linear relationship between axial length and elp is dereived
            Parameters:
            -----------
            Aconst- float,
                A-constant for the IOl
            axialLength- float
                axial length in mm
            meanK - float
                average keratometry (D)
            Rt- float
                target refraction (D)
            Output:
            ----------
            IOL power
        """
        # predict the elp
        # first, compute the ACD_const
        # ACD_const = 0.58357*Aconst - 63.896 # (mm)
        ACD_const = 0.62467*Aconst-68.747


        elp = ACD_const +0.292*axialLength-6.87
        # elp = 0.292*axialLength - 2.93
        # predict IOL power at spectacle plane
        P = round(self.ThinLensPower(elp,meanK,axialLength,Rt/(1+v*Rt))/pDelta)*pDelta
        # predict refraction
        R = self.PredictedRefraction(P,meanK,axialLength,elp)
        # translate back to corneal plane
        R = R/(1-v*R)
        return P,R,elp

    def HofferQ(self,meanK,axialLength,pACD,Rt,v=0.012,n_v=1.336,n_c=1.336,pDelta=0.5,rDelta=None):
        """
            Compute the Hoffer Q predicted IOL power and ACD
            Based on:
            Hoffer, Kenneth J.
            "The Hoffer Q formula: a comparison of theoretic and regression formulas."
            Journal of Cataract & Refractive Surgery 19.6 (1993): 700-712.

            Parameters:
            ----------
            Aconst, float
                the IOL manufacturere A-constant
            v, float, default =0.012
                vertex distance (m)
            axialLength, float
                axial length (mm)
                if axialLength is not provided or is None, it will
                be evaluated using the Zheng approximation
            Rt, float
                target refraction (D)
            pACD- float
                the average individualized ACD estimated from retrospective data
                see method HofferQPersonalisedACD()

            Output:
            -------
            P, float
                predicted IOL power (D)
            Rs, float
                predicted refraction (D), sphere only
        """

        if isinstance(meanK,pd.Series):
            meanK = meanK.values
        if isinstance(axialLength,pd.Series):
            axialLength = axialLength.values
        if isinstance(Rt, pd.Series):
            Rt = Rt.values

        # impute missing values
        A    = np.where(np.isnan(axialLength)|(axialLength<=0),self.K2Al(meanK),axialLength)
        A    = np.where(A>31,31,A)
        A    = np.where(A<18.5,18.5,A)
        # A    = 1000*n_c/(P + (1.336/(1.336/(meanK + R)- C + 0.05)/1,000)))) + C + 0.05
        M    = np.where(axialLength<=23,1,-1)
        G    = np.where(axialLength<=23,28,23.5)
        # personalized ACD
        pACD = np.where(pACD>6.5,6.5,pACD)
        pACD = np.where(pACD<2.5,2.5,pACD)

        # pACD = self.HofferQPersonalisedACD()
        elp  = pACD + 0.3*(A-23.5)+np.tan(meanK)**2 +\
               (0.1*M*((23.5-A)**2)*np.tan(0.1*(G-A)**2))-0.99166 # in mm
        elp = np.where(elp>6.5,6.5,elp)
        elp = np.where(elp<2.5,2.5,elp)
        elp+=0.05

        # predict IOL power
        P = self.ThinLensPower(elp,meanK,A,Rt,n_c=n_c,n_v=n_v,pDelta=pDelta)
        R = self.PredictedRefraction(P,meanK,A,elp,n_c=n_c,n_v=n_v,rDelta=rDelta,v=v)

        # A = 1336/(P+(1.336/(1.336/(43.5+R) -(4.5+0.05)/1000))) +4.5+0.05

        return P,R,elp,A

    @staticmethod
    def HofferQPersonalisedACD(meanK,al,Pi,Rf,n_c=1.336,v=0.012):
        """
            Estimate the personalised ACD used for Hoffer-Q formula based on retrospective data analysis

            Parameters:
            -----------
            meanK, float
              mean keratometry (D)

            al, float
             axial length (mm)
            v, float, default = 0.012
              vertex distance (m)
            n_c, optional, default = 1.336
              refractive index cornea
            Pi, float
              implanted power (D)
            al, float
              axial length(m)

            Output:
            -------
            pACD,
             personalised ACD (m)

        """

        # make sure input parameters are np arrays
        if isinstance(meanK, pd.Series):
            meanK = meanK.values
        if isinstance(al,pd.Series):
            al = al.values
        if isinstance(Pi, pd.Series):
            Pi = Pi.values
        if isinstance(Rf, pd.Series):
            Rf = Rf.values
        # n_c  = 1.336
        # l    = np.asanyarray(self.opDay['IolAxialLength_Left'].append(self.opDay['IolAxialLength_Right']))
        # K    = self.opDay['IolMeanK_Left'].append(self.opDay['IolMeanK_Right'])
        # al   = self.data['l_axial_length_mean'].append(self.data['r_axial_length_mean'])
        # r1   = self.data['l_radius_r1'].append(self.data['r_radius_r1'])
        # r2   = self.data['l_radius_r2'].append(self.data['r_radius_r2'])
        # r    = 0.5*(r1+r2)
        # meanK    = 337.5/r
        # Pi   = self.data['IolPower_Left'].append(self.data['IolPower_Right'])
        # Rf   = self.data['IolFinalRefraction_Left'].append(self.data['IolFinalRefraction_Right'])

        # Rrm  = Rr/(1-v*Rr) # translate to cornea plane
        lInds = np.where( ~(np.isnan(meanK) & (meanK<35) & \
                           np.isnan(Pi)    & (Pi==0)   & \
                           np.isnan(al)    & (al<15)    & \
                           np.isnan(Rf)))[0]

        meanK = meanK[lInds]
        Pi    = Pi[lInds]
        A     = al[lInds]
        Rf    = Rf[lInds]
        R     = Rf/(1-v*Rf) # corneal plane
        N     = n_c/(meanK+R)

        Delta = (A-N)**2 + 4*n_c*(N-A)/Pi
        lInds = np.where(Delta>=0)[0]

        pACD = 0.5*(A[lInds]+N[lInds]-(Delta[lInds])**0.5 ) - 0.05
        # keep on valid entries
        return np.mean(pACD)*100 # (m)

    def Shammas(self,Aconst,meanK,axialLength,Rt,n_v=1.333,n_c=1.3333,v=0.012,pDelta=0.5,rDelta=None):
        """
            Compute IOL power according to SHammas's formula.

            Parameters:
            -----------
             Aconst - float,
                 lens A-constant
             meanK - float
                 average keratometry (D)
             axialLength, float
              axial lenth (mm)
             Rt, float
               target refaraction (D)
             n_v, float, default = 1.336
              refractive index vtreous
             n_c, float, default = 1.333
              refractive index of the cornea
             v, float, default = 0.012
               vertex distance (m)
             pDelta, float, default =0.5
              diptric interval to round computed power
             rDelta, float default=None
               diopteric delta to round predicted refraction
               if set to None, no rounding is performed

            Returns:
            -----------
             P, float
              iol power (D)
             R, float
              expected refraction based on rounded P (D)
             elp, float
              expected lens positionm
             al, float
              axial length (mm)
        """

        if not isinstance(Aconst, (float,int,pd.Series,np.ndarray)):
            raise ValueError(f'Aconst must be numerical array, got {Aconst.__class__}')
        if not isinstance(meanK,(float,int,pd.Series,np.ndarray)):
            raise ValueError(f'meanK must be numerical array, got {meanK.__class__}')
        if not isinstance(axialLength,(float,int,pd.Series,np.ndarray)):
            raise ValueError(f'axialLength must be numerical array, got {axialLength.__class__}')

        if isinstance(axialLength,pd.Series):
            axialLength = axialLength.values
        if isinstance(meanK,pd.Series):
            meanK = meanK.values
        if isinstance(Rt, pd.Series):
            Rt = Rt.values

        al = np.where(np.isnan(axialLength),self.K2Al(meanK),axialLength)
        al = 0.9*al+2.3 # correction to the axial length (mm), see Shammas 1982
        # predict ELP (mm)
        elp  = (0.5835*Aconst-64.4 -0.05)*np.ones(len(al))
        k    = (1.14*meanK-6.8)/1.0125 # corrected keratometry
        # compute IOL power  in spectacle plane
        P    = self.ThinLensPower(elp,k,al,Rt/(1+v*Rt),n_c=n_c,n_v=n_v,pDelta=pDelta)

        # Predict refraction at cornea plane
        R = self.Refraction('shammas',P,k,elp,al,n_c=n_c,v=v,rDelta=rDelta)
        return P,R,elp,al

    def HolladayI(self,Aconst,meanK,axialLength,Rt,sFactor=None,v=0.012,n_v=1.336,n_c=1.333,pDelta=0.5,rDelta=None):
        """
            Estimate IOL power using the Hollady I formula

            Parameters:
            -----------
            Aconst- float
                the IOL A-constant from manufacturer
            meanK - float
                average keratometry (D)
            axialLength- float
                axial length in mm
            Rt- float
                target refraction (D)

            Returns:
            -----------
        """
        # make sure input parameters are numpy arrays
        if isinstance(meanK,pd.Series):
            meanK = meanK.values
        if isinstance(axialLength,pd.Series):
            axialLength = axialLength.values
        if isinstance(Rt,pd.Series):
            Rt = Rt.values


        Rc          = self.K2Rc(meanK,n_v=1.3375)*1000 # m
        # complete missing values
        axialLength = np.where(np.isnan(axialLength),self.Rc2Al(Rc),axialLength)
        al          = axialLength+0.2
        ag          = 12.5*axialLength/23.45
        ag          = np.where(ag>13.5,13.5,ag)

        # Compute radius of cornea
        rm  = self.K2Rc(meanK)*1000 # units of mm
        rm  = np.where(rm<7,7,rm)   # truncate
        # Compute the AACD
        aACD   = 0.56 +rm-(rm**2 - (ag**2)/4)**0.5
        # compute the surgeon factor
        if sFactor==None  or np.isnan(sFactor):
            s = sFactor
        else:
            s = self.Aconst2SurgeonFactor(Aconst)# 0.5663*Aconst-65.6

        # compute the elp
        elp = aACD+s #mm

        # compute IOL power
        Ap  = 1000*n_v*(n_v*Rc-(n_c-1)*al-0.001*Rt*(v*(n_v*Rc-(n_c-1)*al)+al*Rc))
        Bp  = (al-elp)*(n_v*Rc-(n_c-1)*elp-0.001*Rt*(v*(n_v*Rc-(n_c-1)*(elp))+(elp)*Rc))
        if pDelta is not None:
            P   = np.round((Ap/Bp)/pDelta)*pDelta
        else:
            P   = (Ap/Bp)
        # Ar  = 1000*n_a*(n_a*Rc-(n_c-1)*al)-P*(al-elp)*(n_a*Rc-(n_c-1)*elp)

        # compute expected refraction at corneal plane
        Ar  = 1000*n_v*(n_v*Rc-(n_c-1)*al)-P*(al- elp)*(n_v*Rc-(n_c-1)*(elp))
        Br  = n_v*(v*(n_v*Rc-(n_c-1)*al)+al*Rc)-0.001*P*(al-elp)*(v*(n_v*Rc-(n_c-1)*(elp))+(elp)*Rc)
        R   = Ar/Br
        R   = R/(1-v*R) # translate to tcorneal plane
        if rDelta is not None:
            R = np.round(R*rDelta)/rDelta
        # R = R/(1+v*R) # translate back to corneal plane

        # Identify atypical moncular results
        Pn   = self.ThinLensPower(elp,43.81,23.5,0)
        flag = np.where(((axialLength*1000)<22) | ((axialLength*1000)>25),False,True)
        flag = np.where((meanK<40) | (meanK>47),False,flag)
        flag = np.where(np.abs(P-Pn)>3,False,flag)
        return P,R,elp,al,flag

    def HolladaySurgeonFactor(self,meanK,al,Pi,Rf, v=12):
        """
            Use post operative data to estimate the surgeon factor for Holladay I formula
            Parameters:
            ----------
            Pi- float,
                implanted IOL power (D)
            Rr- float
                resulted refractive error post-op (D)
            al - float
                modified pre-surgery axial length (m)
            b - float
                vertex distance between spectacles and cornea (m)
            rm- float
                pre-surgery mean corneal radius of curvature (m)
            n_a- float
                refractive index of the aqueous
            n_e - float, default=4/3
                refractive index of the cornea
        """

        # discard missing values
        inds = np.where( Pi.notna()\
                         & (Pi!=0) \
                         & (Rf.notna()) \
                         & meanK.notna()\
                         & (meanK>0) \
                         & al.notna()\
                         & (al>0))[0]
        Pi    = Pi.iloc[inds].values
        Rf    = Rf.iloc[inds].values
        meanK = meanK.iloc[inds].values
        al    = al.iloc[inds].values

        Rc   = self.K2Rc(meanK)*1000
        # Rc   = r.iloc[inds].values
        rg   = np.where(Rc<7,7,Rc)
        ag   = 12.5*al/23.45
        ag   = np.where(ag>13.5,13.5,ag)
        # compute the AACD
        acd  = 0.56 + rg -(rg**2 -(ag**2)/4)**0.5
        al  +=  0.2     # corrected pre-op data
        nc   = 4/3      # refractive index cornea
        na   = 1.336    # refractive index aqueous

        aq   = (nc-1) - (0.001*Rf*((v*(nc-1)) - Rc))
        bq   = Rf*0.001*((al*v*(nc-1))-(Rc*(al-(v*na))))-(((nc-1)*al) + (na*Rc))
        cq1  = 0.001*Rf*((v*((na*Rc)-((nc-1)*al)))+(al*Rc))
        cq2  = (1000*na*((na*Rc)-((nc-1)*al)-cq1))/Pi
        cq3  = (al*na*Rc)-(0.001*Rf*al*v*Rc*na)
        cq   = cq3-cq2
        desc = ((bq**2)-(4*aq*cq))
        desc = np.where(desc<0,0,desc)
        si   = (((-bq)-desc**0.5 )/(2*aq)) - acd

        return si[np.isnan(si)==False].mean()

    def PredictedRefraction(self,Pi,meanK,al,elp,n_c=1.336,n_v=1.337,v=0.012,rDelta=None):
        """
         Reverse Gaussian optics formula to retrieve refraction

         Parameters:
         -----------
         Pi- float
             IOL power (D)
         meanK, float
            average keratometry (D)
         al, float
            axial length (mm)
         elp, float
            postoperative anterior chamber depth (mm)
         v, vertex distance,float, default=0.012, (m)
         n_v, float, default=1.337
          refractive index acquose medium
         n_c, float, default = 1.336
          refractive index cornea
         Output:
         -------
          Rs, float
            predicted refraction (sphere only) (D)
        """
        # compute the corneal radius
        # Rc  = self.K2Rc(meanK,n_v=n_a) # m
        # Ar  = n_a*(n_a*Rc-(n_c-1)*al)-Pe*(al-elp)*(n_a*Rc-(n_c-1)*elp)
        # Br  = n_a*(v*(n_a*Rc-(n_c-1)*al)+al*Rc)-Pe*(al-elp)*(v*(n_a*Rc-(n_c-1)*elp)+elp*Rc)
        # R   = Ar/Br

        R = n_c/(n_c/(n_v/((al-elp)/1000) -Pi) +elp/1000) -meanK
        R = R/(1-v*R) # return to corneal plane
        if rDelta is not None:
            return np.round(R/rDelta)*rDelta
        else:
            return R

    def ReviseAconst(self,Aconst,meanK,al,Pi,se):
        """
         Re-compute the A constant based on postoperative results
         Based on the text:
         Barrett, Graham D.
         "Intraocular lens calculation formulas for new intraocular lens implants."
         Journal of Cataract & Refractive Surgery 13.4 (1987): 389-396.

         Parameters:
         -----------
        """
        Cr = 1/(0.875*Aconst-8.55)
        Dp = Aconst-2.5*al-0.9*meanK-se/Cr # difference in power
        return Aconst +(Pi-Dp) # the new Aconst

    def Refraction(self,formula,P,meanK,elp,al,n_c=1.333,n_v=1.336,v=0.012,rDelta=None):
        """
         Refraction based on power implanted (P) keratomaetry (meanK), effective lens position (elp)
         and axial length (al)
         results in D
        """
        if formula.lower() in ['srkt','t2','holladay-1']:
            rc = (n_c-1)/meanK
            Ar = n_v*(n_v*rc-(n_c-1)*al*0.001)-P*0.001*(al-elp)*(n_v*rc-(n_c-1)*elp*0.001)
            Br = n_v*(v*(n_v*rc-(n_v-1)*al*0.001)+al*rc*0.001)-0.001*P*(al-elp)*(v*(n_v*rc-(n_c-1)*elp*0.001)+elp*rc*0.001)
            R  = Ar/Br # is it in the corneal or spectcle plane?
            R  = R/(1-v*R)
            if rDelta is not None:
                R = np.round(R/rDelta)*rDelta
        elif formula.lower() in ['barrett-1','haigis','binkhorst-2','hoffer-q','olsen']:
            R = self.PredictedRefraction(P,meanK,al,elp,n_c=n_c,n_v=n_v,v=v,rDelta=rDelta)
        elif formula.lower() in ['shammas']:
            rc  = 1000*(n_c-1)/meanK
            Ar  = 1000*n_c*(4*rc - al) - P*(al - elp)*(4*rc - elp)
            Br  = 1000*n_c*(v*(4*rc - al) + 0.003*al*rc)- P*(al - elp)*(v*(4*rc - elp) + 0.003*elp*rc)
            R   = Ar/Br


        else:
            print(f'{formula} not in list of formulas')
        return R