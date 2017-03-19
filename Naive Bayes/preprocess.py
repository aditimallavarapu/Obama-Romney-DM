# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:14:39 2017

@authors: Aditi and Suganya
"""

import os
import re
import string
import xlrd
from nltk.corpus import stopwords



class Preprocess:
    def cleanup(self, data):
        cleantext = data.replace(",","")        #remove commas
        cleaner = re.compile('<.*?>')           #remove tags
        cleantext= re.sub(cleaner,' ', cleantext)        
        ascii = set(string.printable) 
        cleantext=filter(lambda x: x in ascii , cleantext)   #remove emoji and non english characters
        cleantext= re.sub(r'https?:\/\/.*[\r\n]*', '', cleantext)       #remove links
        cleantext = cleantext.translate(string.maketrans("",""), string.punctuation)
        cleantext = cleantext.translate(None, string.digits)
        stop = set(stopwords.words('english')) - set(('and', 'or', 'not'))
        cleantextlist = [i for i in cleantext.lower().split() if i not in stop]      #remove stopwords except few exceptions  
        cleantext = ' '.join(cleantextlist)
        return cleantext
        

    def xls_to_txt(self, filename,obama,romney):
        x =  xlrd.open_workbook(filename)#, encoding_override = "utf-8")
        x1 = x.sheet_by_index(0)        
        x2 = x.sheet_by_index(1)
        
        obama_file = open(obama, 'wb')
        for rownum in xrange(3,x1.nrows):
            obama_file.write(u'\t'.join([i if isinstance(i, basestring) else str(int(i)) for i in x1.row_values(rownum, 3, 5)]).encode('utf-8').strip()+ "\t\n")
        obama_file.close()
        
        romney_file = open(romney, 'wb')
        for rownum in xrange(3,x2.nrows):
            romney_file.write(u'\t'.join([i if isinstance(i, basestring) else str(int(i)) for i in x2.row_values(rownum, 3, 5)]).encode('utf-8').strip()+ "\t\n")
        romney_file.close()
        
    def clean_text_files(self,romney_write,romney_read,obama_write,obama_read):
        romney_file = open(romney_write, 'w')
        obama_file = open(obama_write, 'w')
        rel_path = romney_read
        script_dir= os.path.dirname(os.path.abspath(__file__))
        abs_file_path = os.path.join(script_dir, rel_path)
        f = open ( abs_file_path )
        for line in f.readlines():
            cols = line.split("\t")
            cols[0] = self.cleanup(cols[0]) 
            romney_file.write(cols[0])
            romney_file.write("\t")
            romney_file.write(cols[1])
            romney_file.write("\n")
        romney_file.close()
        rel_path = obama_read
        abs_file_path = os.path.join(script_dir, rel_path)
        f = open ( abs_file_path )
        for line in f.readlines():
            cols = line.split("\t")
            cols[0] = self.cleanup(cols[0]) 
            obama_file.write(cols[0])
            obama_file.write("\t")
            obama_file.write(cols[1])
            obama_file.write("\n")
        obama_file.close()
        f.close()
            
