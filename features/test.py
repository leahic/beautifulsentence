# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import features

def test(obj , attrname):
	if getattr(obj , attrname , None) is None:
		print attrname , 'test failed'
		return False
	else:
		print attrname , 'test pass'
		return True


obj = features.Feature(datapath ='../data')
obj.read('sample_sentence.txt')

TESTLIST = [
		'feature1',
		'feature2',
		'feature3',
		'feature4',
		'feature5',
		'feature6',
		'feature7',
		'feature8',
		'feature9',
		'feature10',
		'feature11',
		'feature12'
	      ]	

for attrname in TESTLIST:
	getattr( obj , 'solve_' + attrname )()
obj.save('featureobj')

flag = True

print ""
print "TEST:"
print ""

obj2 = features.Feature(datapath ='../data')
obj2 = obj2.load('featureobj')
for attrname in TESTLIST:
	flag = test(obj2 , attrname) and flag

#print getattr( obj , TESTLIST[-1] ) 

if flag:
	print "all test pass"
else:
	print "some errors"