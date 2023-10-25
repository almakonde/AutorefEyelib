from autorefeyelib.Parsers import Oplus
from autorefeyelib.Parsers import vx120

class ParseMerge:

    def Run(self,oplusFile, vxFolder):
        print('[ParseMerge] Parsing oplus file')
        o = Oplus.Parser()
        o.Load(oplusFile)
        oplusDB = o.Parse()

        print('[ParseMerge] Parsing vx folders')
        vParser = vx120.Parser()
        vxData  = vParser.BatchParse(vxFolder)

        print('[ParseMerge] Mergin Oplus data with  vx120')



