# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import sys
import time
from collections import defaultdict
import jieba
import jieba.posseg as pseg
import gensim
import numpy as np
import utils

class Feature(utils.SaveLoad):

	def __init__(self , datapath = '../data'):
		self.datapath = datapath if datapath.endswith('/') else datapath + '/'
		jieba.setLogLevel('NOTSET')

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

	def solve_feature11(self , redo = False):
		if getattr(self , 'feature11' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature10()
			np.seterr(all = 'ignore')
			self.feature11 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature10[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature11.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature12(self , redo = False , vecsize = 10):
		if getattr(self , 'feature12' , None) is None or redo:
			if vecsize < 2:
				raise('vecsize is less than 2')

			vecsize -= 1

			propertydict = defaultdict(int)

			for doc in self.docs_norm:
				for sentence in doc:
					for word in sentence:
						propertydict[ utils.property(word) ] += 1

			propertydict = sorted( [  (word , propertydict[word])  for word in propertydict ] , key = lambda x : - x[1] )

			propertyindex = defaultdict(int)

			vecsize -= 1

			for index , (word , value) in enumerate(propertydict):
				propertyindex[word] = index if index < vecsize else vecsize

			vecsize += 1

			self.feature12 = []
			self.feature12_len = vecsize + 1

			for doc in self.docs_norm:

				docvec = []

				for sentence in doc:
					length = float( len(sentence) )
					propertyvec = [0.0] * vecsize
					propertyvec = np.array(propertyvec)

					for word in sentence:
						propertyvec[ propertyindex[ utils.property(word) ] ] += 1.0
					
					propertyvec  = propertyvec / length

					entropy = np.sum (  [ - x * np.log2(x) for x in propertyvec if not x == 0] )

					propertyvec = list(propertyvec)
					propertyvec.append( entropy )

					docvec.append(  propertyvec  )

				self.feature12.append( docvec )

	


