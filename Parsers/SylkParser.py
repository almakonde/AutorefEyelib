import pandas as pd

class Parser:
	"""
		A simple parser for .slk files
	"""
	def Parse(self,filename, encoding='utf-8',delimiter=';',endofLine="\n"):
		"""
			Parse a sylk database into a dataframe
		"""
		fid    = open(filename,'r',encoding=encoding)
		lines  = fid.readlines()
		res    = pd.DataFrame()
		for lIdx in lines:
		# format C;Xi;Yj;value
		# first build a table with the right number of columns
			lSplit = lIdx.split(delimiter)
			if len(lSplit)==4:
				# get columns position
				col = int(lSplit[1].replace('X',''))
				row = int(lSplit[2].replace('Y',''))
				val = lSplit[3].lstrip('K').rstrip("\n").replace("\"","") # remove leading K
				if row==1:
					res.loc[0,val] = None
				else:
					res.loc[row-2,res.keys()[col-1]] = val
		return res





