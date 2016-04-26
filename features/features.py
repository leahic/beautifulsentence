# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import sys
import time
import jieba
import jieba.posseg as pseg
import gensim
import numpy as np
import utils

class Feature(utils.SaveLoad):

	def __init__(self , datapath = '../data'):
		self.datapath = datapath if datapath.endswith('/') else datapath + '/'

	def read(self , fname , redo = False):
		if getattr(self ,  'docs' , None) is None or redo:
			f = file(self.datapath + fname , 'r')
			self.docs = []
			doc = []
			for line in f:
				if line.startswith('********************'):
					self.docs.append(doc)
					doc = []
				else:
					doc.append(line.strip('\n').decode('utf-8'))	

		if getattr(self , 'docs_norm' , None) is None or redo:
			punctuation = [u'，' , u'。' , u'…'  , u'；' , u'：' , u'“' ,u'”', u'、' ,  u'+' , u'-' , u'！' , u'—' , u'《' , u'》' , u'？' , u'（' , u'）'  ,u'·' , u'!' ,u'’' ,u'‘' ,u'.' ,u',']
			punctuation.append('')		
			self.docs_norm = [ [ [ word for word in sentence.split(' ') if word not in punctuation ] for sentence in doc ] for doc in self.docs ]			

	def readstopwords(self , redo = False):
		if getattr(self , 'stopwords' , None) is None or redo:
			f = file(self.datapath + 'stopwords.txt','r')
			self.stopwords = [ line.strip('\n').decode('utf-8') for line in f ]
			f.close()

	def readassociated(self , redo = False):
		if getattr(self , 'associated' , None) is None or redo:
			f = file(self.datapath + 'associated_words.txt','r')
			self.associated = [ line.strip('\n').decode('utf-8') for line in f ]
			f.close()		

	def solve_feature1(self , redo = False):
		if getattr(self , 'feature1' , None) is None or redo:
			self.feature1 = [ [ len( ''.join(sentence) ) for sentence in doc ] for doc in self.docs_norm ]
 
	def solve_feature2(self , redo = False):
		if getattr(self , 'feature2' , None) is None or redo:
			self.feature2 = [ [ len( sentence ) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature3(self , redo = False):
		if getattr(self , 'feature3' , None) is None or redo:
			self.solve_feature1()
			self.solve_feature2()
			np.seterr(all = 'ignore')
			self.feature3 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature1[index] , dtype = np.float32)
				y  =  np.array( self.feature2[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature3.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature4(self , redo = False):
		if getattr(self , 'feature4' , None) is None or redo:
			self.feature4 = [ [ len(set(sentence)) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature5(self , redo = False):
		if getattr(self , 'feature5' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature4()
			np.seterr(all = 'ignore')
			self.feature5 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature4[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature5.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature6(self , redo = False):
		if getattr(self , 'feature6' , None) is None or redo:
			self.readstopwords()
			self.feature6 = [ [ len(set( [ word for word in sentence if word not in self.stopwords] )) for sentence in doc ] for doc in self.docs_norm ]


	def solve_feature7(self , redo = False):
		if getattr(self , 'feature7' , None) is None or redo:
			self.solve_feature4()
			self.solve_feature6()
			np.seterr(all = 'ignore')
			self.feature7 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature4[index] , dtype = np.float32)
				y  =  np.array( self.feature6[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature7.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature8(self , redo = False):
		if getattr(self , 'feature8' , None) is None or redo:
			self.readstopwords()
			self.feature8 = [ [ len(set( [ word for word in sentence if word not in self.stopwords and len(word) == 4 ] )) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature9(self , redo = False):
		if getattr(self , 'feature9' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature8()
			np.seterr(all = 'ignore')
			self.feature9 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature8[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature9.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature10(self , redo = False):
		if getattr(self , 'feature10' , None) is None or redo:
			self.readassociated()
			self.feature10 = [ [ len( [ word for word in sentence if word in self.associated ] ) for sentence in doc ] for doc in self.docs_norm ]