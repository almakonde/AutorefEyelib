import os
import xml.etree.ElementTree as ET
import pandas as pd

class Parser:
    """
        Parse an xml file to pandas.Series or DataFrame
    """

    def Parse(self,filePath,index=None):
        """
            Parse an xml to pandas Series or Dataframe

            Parameters:
            -----------
            filePath : str
                a path to the xml file to parse
            index : {str,int, float}, default=None
                the index to assign to a row of the dataframe output
                if index=None, the output will be pandas.Series

            Output
            --------
            results : pandas.Series, or DataFrame
                 parsed xml file into a Series
                 keys of the resulting series correspond to depth of the xml tree
                 e.g. key = parentField_childField_grandchildField
                 if input index=None, the resulting parsed xml will be a series

            Raises
            --------
            ValueError

        """
        if filePath is None:
            # Exit returning None
            return None

        if filePath.__class__ is not str:
            raise ValueError(f'filePath must be a string got {filePath.__class__}')

        if not os.path.exists(filePath):
            raise ValueError(f'Cannot locate the file {filePath}, please verify the file is in the path specified')

        if not filePath.endswith('.xml'):
            raise ValueError(f'file must be an xml, got {filePath}')


        print(f'[Info][xmlParser] Started parsing file {filePath}')

        xmlFile  = ET.parse(filePath)
        root     = xmlFile.getroot()

        def inspectNode(node,d,lastTag):
            """
                Internal function for recrussively traverse the xml tree.
                The keys in each successive levels are appended underscore (_)
                values are transfored to float when possible

                Parameters
                -----------
                node : ElementTree.Element
                    the current nose to inspect
                d : dictionary
                    the current dictionaty, values extracted from the
                    node are appended to the dictionary
                lastTag : string
                    the tag of the parent of the current node

                Output
                -------
                d : dictionary
                    updated dictionay with appended values and corresponding keys

                Raises
                --------
                 valueError
            """
            nTag = node.tag if lastTag=='' else lastTag+'_'+node.tag
            if len(node)>0:
                for nIdx in node:
                    d = inspectNode(nIdx,d,nTag)
            else:
                val = node.text
                try:
                    val = float(val)
                except:
                    pass
                d.update({nTag:val})
            return d

        d = {}
        # start recursion from first level
        d = inspectNode(root[0],d,'')
        results = pd.DataFrame().from_dict(d,orient='index')
        if index!=None:
            results = results.T
            results.index = [index]
        print(f'[Info][xmlParser] Done')
        return results



