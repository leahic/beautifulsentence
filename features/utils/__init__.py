# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import sys
import time
import numpy as np
import jieba
import jieba.posseg as pseg
import cPickle as pickle

jieba.setLogLevel('NOTSET')

def property(val):
	if isinstance(val , str) or isinstance(val , unicode):
		cutlist = pseg.cut(val , HMM=False)
		return ''.join( [ flag for word , flag in cutlist ] )
	else:
		raise("param must be unicode")

def scalemax(vec):
	vec = np.array(vec)
	if vec.max() != 0:
		vec =  vec / vec.max()
	return vec

def normalize(vec):
	vec = np.array(vec)
	denominator = np.sum( vec )
	if  denominator != 0:
		vec  = vec / denominator
	return list(vec)

class SaveLoad(object):
	def load(self, fname):
		f = file(fname,'rb')
		obj = pickle.load(f)
		f.close()
		return obj

	def save(self, fname):
		f = file(fname , 'wb')
		pickle.dump(self, f)
		f.close()