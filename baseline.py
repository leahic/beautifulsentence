# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

from features import features , utils
import numpy as np

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


def MAP(matrix , labels):

	finalmap = []

	for index in range(len(matrix)):
		result = [ (matrix[index][index2] , labels[index][index2]) for index2 in range(len(matrix[index])) ]
		result = sorted(result , key = lambda x : -x[0])
		themap = []
		nums = 0.0
		for rank , (value , label) in enumerate(result):
			if label > 0:
				nums += 1.0
				themap.append( nums / float(rank + 1) )
		if np.isnan( np.average(themap) ):
			finalmap.append( 0.0 )
		else:
			finalmap.append(np.average(themap))

	result = np.average( finalmap )
	if np.isnan(result):
		return 0.0
	return result

def MRR(matrix , labels):

	finalmrr = []

	for index in range(len(matrix)):
		result = [ (matrix[index][index2] , labels[index][index2]) for index2 in range(len(matrix[index])) ]
		result = sorted(result , key = lambda x : -x[0])
		mrr = 0.0
		for rank , (value , label) in enumerate(result):
			if label > 0:
				mrr = 1.0 / float(rank + 1)
				break
		finalmrr.append( mrr )

	result = np.average( finalmrr )
	if np.isnan(result):
		return 0.0
	return result

def PR(matrix , labels , k):
	prvec = [ [] , [] ]

	for index in range(len(matrix)):
		result = [ (matrix[index][index2] , labels[index][index2]) for index2 in range(len(matrix[index])) ]
		result = sorted(result , key = lambda x : -x[0])
		target = result[:k]

		molecular =  float(len([ label for (value , label) in target if label > 0 ]))
		denominator = float(len([ label for (value , label) in result if label > 0 ]))

		if molecular != 0 and denominator != 0:
			precision =  molecular / float(len(target))
			recall = molecular / denominator
		else:
			precision = 0.0
			recall = 0.0

		prvec[0].append( precision  )
		prvec[1].append( recall  )

	finalpr = [ np.average(prvec[0]) , np.average(prvec[1]) ]
	return finalpr




def analysis(matrix , labels , name):
	f = file('baseline/' + name + '_analysis.txt','w')
	data = [
  		MAP(matrix,labels),
  		MRR(matrix,labels),
  		PR(matrix,labels,1),
  		PR(matrix,labels,5),
  		PR(matrix,labels,10)
		]
	f.write( str(data)  + '\n')
	f.close()

	global origin
	pastline = 0

	f = file('baseline/' + name + '_sentence.txt','w')
	for index in range(len(matrix)):
		ranks = enumerate(matrix[index])
		ranks = sorted(ranks , key = lambda x : -x[1])

		for index2 , value in ranks:	
			f.write(origin[index2 + pastline].encode('utf-8') + '\n' )
		f.write('********************\n')

		pastline += len(matrix[index])

	f.close()


def solve_zhuangya(obj , labels):

	zhuangya = []

	for index in range(len(obj.docs_limit)):
		docvec = []
		for index2 in range(len(obj.docs_limit[index])):
			value = ( obj.feature21[index][index2] + obj.feature23[index][index2] + 
			            obj.feature14[index][index2][3] + obj.feature9[index][index2] ) / 4.0			
			docvec.append(value)
		zhuangya.append(docvec)
	analysis(zhuangya , labels ,  'zhuangya')

def solve_like(obj , labels):

	like = [ utils.scalemax(docvec) for docvec in obj.feature30]
	analysis(like , labels ,  'like')


def solve_parallel(obj , labels):
	
	parallel = [ utils.scalemax(docvec) for docvec in obj.feature31]
	analysis(parallel , labels ,  'parallel')


def solve_textrank(obj , labels):

	textrank = obj.feature37
	analysis(textrank , labels ,  'textrank')

def main():

	obj = features.Feature().load('features/featureobj')
	global origin
	f = file('data/sample_sentence.txt','r')
	origin = [line.strip('\n').decode('utf-8').replace(' ', '') for line in f if not line .startswith('********************')]
	f.close()

	print len(origin)

	labels = getlables()

	solve_zhuangya(obj , labels)

	solve_like(obj , labels)

	solve_parallel(obj , labels)

	solve_textrank(obj , labels)



if __name__ == '__main__':
	main()