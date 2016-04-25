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


obj = features.Feature()
obj.read('../data/sample_sentence.txt')
obj.solve_feature1()
obj.solve_feature2()
obj.save('featureobj')

flag = True

print ""
print "TEST:"
print ""

obj2 = features.Feature()
obj2 = obj2.load('featureobj')
flag = test(obj2 , 'docs') and flag
flag = test(obj2 , 'feature1') and flag
flag = test(obj2 , 'feature2') and flag
print obj2.feature2

if flag:
	print "all test pass"
else:
	print "some errors"