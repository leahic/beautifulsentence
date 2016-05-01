# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import os
from features import features


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


def main( options = ['-c' , '100'] , datapath = 'data' , programpath = 'svmrank' ,traindat = 'train.dat' , modeldat = 'model.dat' , pnrate = 1.0):
	if not datapath.endswith('/'):
		datapath = datapath + '/'
	if not programpath.endswith('/'):
		programpath = programpath + '/'

	obj = features.Feature().load('features/featureobj')

	vecs = obj.getvec()
	labels = getlables(datapath)

	f = file(datapath + traindat , 'w')
	for index , doclabel in enumerate(labels):
		for index2 , sentencelabel in enumerate(doclabel):
			if sentencelabel > 0:
				output = '1 qid:' + str(index + 1)
				output = output + ' ' +  ' '.join( [  str(ix + 1) + ':' +  str(value)  for ix ,  value in enumerate(vecs[index][index2]) ] ) + ' #' + str(index) + str(index2)
				f.write( output + '\n')

		negativenum = int(1.0 / pnrate) * len( [ x for x in doclabel if x  > 0 ] )

		nownum = 0

		for index2 , sentencelabel in enumerate(doclabel):
			if sentencelabel == 0:
				nownum += 1
				if nownum > negativenum:
					break
				output = '0 qid:' + str(index + 1)
				output = output + ' ' +  ' '.join( [  str(ix + 1) + ':' +  str(value)  for ix ,  value in enumerate(vecs[index][index2]) ] )+ ' #' + str(index) + str(index2)
				f.write( output + '\n')
	f.close()

	options = ' '.join(options)

	command = [programpath + 'svm_rank_learn' , options , datapath + traindat , datapath + modeldat ]
	command = ' '.join(command)
	print command
	print "start learning"
	os.system(  command  )
	print "success"

if __name__ == '__main__':
	main()