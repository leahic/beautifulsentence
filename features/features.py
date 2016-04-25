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

	def __init__(self):
		pass

	def read(self , fname , redo = False):
		if getattr(self ,  'docs' , None) is None or redo:
			f = file(fname , 'r')
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

