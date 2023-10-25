class Utils:
    """
     Compute indices related to the corneal topographic map.
     Many of the indices can be found in N. Maeda et al. 1995
    """
    def __init__(self):
        pass
    def SAI(self,topoMap):
        """
         Compute the surface asymmetric index
        """
        pass
    def SRI(self,topMap):
        """
        Compute the surface regularity index. local fluctuations in the corneal power
        centrally weighted and used the data from the first ten rings
        """
        pass
    def DSI(self,topoMap):
        """
         Compute the differential sector index. The average power computed in eight sections of the round topo map
         the DSI reports the greatest difference in the average power between any two sectors
        """
        pass
    def OSI(self,topoMap):
        """
         Compute the maximum difference between the average power between any two opposing slices of the round
         corneal map (8 equal sections)
        """
        pass
    def CSI(self,topoMap):
        """
         Compute the difference in the average area-corrected corneal power between the central area (3mm diameter)
         and an annulus surrounding the central area (3 to 6 mm)
        """
        pass
    def IAI(self,topoMap):
        """
         The irregular astigmatism is the average summation of inter-ring power variation along
         every semi-meridian for the entire analyzed corneal surface
        """
        pass
    def AA(self,topoMap):
        """
         Analyzed area is the interpolated data area to the total analyzed area
        """
        pass
    def SDP(self,topoMap):
        """
         Standard-deviation power is the STD of corneal power array with area correction
        """
        pass