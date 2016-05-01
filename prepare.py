# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

from features import features

def main(datapath = 'data'):
	if not datapath.endswith('/'):
		datapath = datapath + '/'

	obj = features.Feature( datapath )
	print 'reading data'
	obj.read()
	print 'success'
	print 'training'
	obj.solve_all()
	print 'success'
	print 'saving'
	obj.save('features/featureobj')
	print 'success'

if __name__ == '__main__':
	main()