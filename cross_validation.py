# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import os
from features import features
import numpy as np
import random
import time

def getlables(datapath = 'data'):
	if not datapath.endswith('/'):
		datapath = datapath + '/'

	result = []
	f = file(datapath + 'LABLES.txt','r')
	for line in f:
		line = line.split()
		line = [eval(x) for x in line]
		result.append(line)
	f.close()
	return result

def makedirs(programpath , k_fold):
	global nowtime

	if not programpath.endswith('/'):
		programpath = programpath + '/'

	try:
		os.mkdir(programpath + nowtime)
	except OSError , e:
		print "check the local time setting and the dirs"
		raise(e)
	for groupId in range(1 , k_fold + 1):
		try:
			os.mkdir(programpath + nowtime + '/' + 'group' + str(groupId) )
		except OSError , e:
			print "check the local time setting and the dirs"
			raise(e)


def outputrain(dirpath , traindat , vecs , labels , trainqid , pnrate):
	if not dirpath.endswith('/'):
		dirpath = dirpath + '/'

	f = file(dirpath + traindat , 'w')
	fqid = file(dirpath + 'trainqid.txt','w')
	for qid in trainqid:
		thevecs = vecs[qid - 1]
		thelabels = labels[qid - 1]

		pnum = len( [label for label in thelabels if label > 0] )
		nnum = int(1.0 / pnrate * pnum)

		positive = []
		negative = []

		for index , label in enumerate(thelabels):
			if label == 1 and len(positive) < pnum:
				positive.append(index)
			if label == 0 and len(negative) < nnum:
				negative.append(index)

		full = sorted(positive + negative)

		for index in full:
			line = [ str( thelabels[index] )  , 'qid' + ':' + str( qid ) ]
			for index2 , value in enumerate(thevecs[index]):
				line.append( str(index2 + 1) + ':' + str(value) )
			line.append('#' + str(qid) + str(index + 1))
			line = ' '.join(line)
			f.write(line + '\n')	
			fqid.write( str(qid) + ' ' +  str(index + 1) + ' ' + str(thelabels[index]) +  '\n' )	
	f.close()
	fqid.close()


def outputest(dirpath , testdat , vecs , labels , testqid , pnrate):
	if not dirpath.endswith('/'):
		dirpath = dirpath + '/'

	f = file(dirpath + testdat , 'w')
	fqid = file(dirpath + 'testqid.txt','w')
	for qid in testqid:
		thevecs = vecs[qid - 1]
		thelabels = labels[qid - 1]

		pnum = len( [label for label in thelabels if label > 0] )
		nnum = int(1.0 / pnrate * pnum)

		positive = []
		negative = []

		for index , label in enumerate(thelabels):
			if label == 1 and len(positive) < pnum:
				positive.append(index)
			if label == 0 and len(negative) < nnum:
				negative.append(index)

		full = sorted(positive + negative)

		for index in full:
			line = [ str( thelabels[index] )  , 'qid' + ':' + str( qid ) ]
			for index2 , value in enumerate(thevecs[index]):
				line.append( str(index2 + 1) + ':' + str(value) )
			line.append('#' + str(qid) + str(index + 1))
			line = ' '.join(line)
			f.write(line + '\n')	
			fqid.write( str(qid) + ' ' +  str(index + 1) + ' ' + str(thelabels[index]) +  '\n' )	
	f.close()
	fqid.close()

def train_predict(programpath , options , dirpath , traindat , testdat , modeldat , predictdat):
	if not programpath.endswith('/'):
		programpath = programpath + '/'
	if not dirpath.endswith('/'):
		dirpath = dirpath + '/'

	command = [ programpath + 'svm_rank_learn' ] + options + [dirpath + traindat] + [dirpath + modeldat]
	command = ' '.join(command)
	print command
	os.system( command )
	print 'trian over'

	command = [ programpath + 'svm_rank_classify' ] + [dirpath + testdat] + [dirpath + modeldat] + [dirpath + predictdat]
	command = ' '.join(command)
	print command
	os.system( command )
	print 'predict over'


def MAP(dirpath , predictdat):
	if not dirpath.endswith('/'):
		dirpath = dirpath + '/'

	targetvecs = []

	f = file(dirpath + predictdat , 'r')
	for line in f:
		targetvecs.append( eval(line) )
	f.close()


	result = []

	qidpos = dict()

	f = file(dirpath + 'testqid.txt','r')
	for linenum , line in enumerate(f):
		qid , index , label= [ eval(x) for x in line.split()]
		if qid not in qidpos:
			result.append( [] )
			qidpos[qid] = len(result) - 1
		result[ qidpos[qid] ].append( [ label , targetvecs[linenum] ] ) 
	f.close()

	mapvec = []

	for index in range(len(result)):
		result[index] = sorted(result[index] , key = lambda x : -x[1])
		themap = []
		for rank , (label , value) in enumerate(result[index]):
			if label > 0:
				themap.append( 1 / float(rank + 1) )

		mapvec.append( np.average(themap) )

	finalmap = np.average( mapvec )

	f = file(dirpath + 'MAP.txt','w')
	f.write( str(finalmap) + '\n')
	f.close()

	return finalmap



def main( options = ['-c' , '100'] , datapath = 'data' , programpath = 'svmrank' ,traindat = 'train.dat' , modeldat = 'model.dat' , testdat = 'test.dat' , predictdat = 'predict.dat' , k_fold = 10.0 , pnrate = 1.0):
	if not datapath.endswith('/'):
		datapath = datapath + '/'
	if not programpath.endswith('/'):
		programpath = programpath + '/'

	global nowtime
	nowtime = time.strftime("%m-%d-%H-%M-%S", time.localtime())
	#read data 
	obj = features.Feature().load('features/featureobj')

	vecs = obj.getvec()

	labels = getlables(datapath)

	#part group
	grouplen = int( np.ceil( float(len(labels)) / k_fold ) )

	fold_group = []

	sequence = set( range(1 , len(labels) + 1) )
	for i in range(int(k_fold)):
		samples = random.sample(sequence , min(len(sequence) , grouplen) ) 
		samples = sorted(samples)
		fold_group.append(samples)
		sequence = sequence - set(samples)

	sequence = set( range(1 , len(labels) + 1) )

	k_fold = int(len(fold_group))
	print k_fold , "groups : " , fold_group

	#make group dirs
	makedirs(programpath , k_fold)

	resultvec = []
	#exec
	for groupId in range(1 , k_fold + 1):
		testqid = sorted( fold_group[groupId - 1] )
		trainqid = sorted( sequence - set(testqid) )

		dirpath = programpath + nowtime + '/' + 'group' + str(groupId)  + '/'

		outputrain(dirpath , traindat , vecs , labels , trainqid , pnrate)
		outputest(dirpath , testdat , vecs , labels , testqid , pnrate)
		train_predict(programpath , options , dirpath , traindat , testdat , modeldat , predictdat)
		resultvec.append( MAP(dirpath , predictdat) )

	print np.average( resultvec )
	f = file(programpath + nowtime + '/' + 'MAP.txt' , 'w')
	f.write( str(resultvec) + '\n')
	f.write( str(np.average(resultvec)) + '\n' )
	f.close()


if __name__ == '__main__':
	main()