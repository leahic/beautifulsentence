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


	def solve_feature1(self , redo = False):
		if getattr(self , 'feature1' , None) is None or redo:
			self.feature1 = [ [ len(sentence.replace(' ','')) for sentence in doc ] for doc in self.docs ]

	def solve_feature2(self , redo = False):
		punctuation = [u'，' , u'。' , u'…'  , u'；' , u'：' , u'“' ,u'”', u'、' ,  u'+' , u'-' , u'！' , u'—' , u'《' , u'》' , u'？' , u'（' , u'）'  ,u'·' , u'!' ,u'’' ,u'‘' ,u'.' ,u',']
		punctuation.append('')
		if getattr(self , 'feature2' , None) is None or redo:
			self.feature2 = [ [ len( [ x for x in sentence.split(' ') if x not in punctuation ] ) for sentence in doc ] for doc in self.docs ]