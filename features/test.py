# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import time
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
		'feature12',
		'feature13',
		'feature14',
		'feature15',
		'feature16',
		'feature17',
		'feature18',
		'feature21',
		'feature22',
		'feature23',
		'feature24',
		'feature27',
		'feature28',
		'feature29',
		'feature30',
		'feature31',
		'feature32',
		'feature36',
		'feature37',
		'feature38',
		'feature39'
	      ]	

print ""
print "TEST:" , time.ctime()
print ""

for attrname in TESTLIST:
	getattr( obj , 'solve_' + attrname )()
obj.save('featureobj')

flag = True



obj2 = features.Feature(datapath ='../data')
obj2 = obj2.load('featureobj')
for attrname in TESTLIST:
	flag = test(obj2 , attrname) and flag

if flag:
	print "all test pass" , time.ctime()
else: 
	print "some errors" , time.ctime()