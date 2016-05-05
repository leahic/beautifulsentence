# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html
import sys

class HtmlPage(dict):

	def __init__(self , title='Sample'):
		super(HtmlPage, self).__init__()
		self.pushit('title',title)
		self.pushit('meta','<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">')
		self.pushit('meta','<meta name="viewport" content="width=device-width, initial-scale=1">')
		self.pushit('script' , '<script src="http://cdn.bootcss.com/jquery/2.2.1/jquery.js"></script>')
		self.pushit('css','<link href="http://cdn.bootcss.com/bootstrap/3.3.4/css/bootstrap.min.css" rel="stylesheet">')		
		self.pushit('script','<script src="http://cdn.bootcss.com/bootstrap/3.3.4/js/bootstrap.js"></script>')		
		self.pushit('css' , '<style> div{ margin-top:50px} </style>')
		self.pushit('nav' , 
				u'''<nav role="navigation" class="navbar navbar-inverse navbar-fixed-top">
				</nav>
				 '''
				)

	def pushit(self , key , value):
		try:
			self[key].append( value )
		except KeyError , e:
			self[key] = [value]


	def wrapit(self , label , value):
		return '<' + label + '>' + value + '</' + label  + '>'

	def getHeader(self):
		result = ['<head>']

		for value in ['meta' , 'title' , 'css' , 'script']:
			try:
				result.append( '\n'.join(self[value]) )
			except KeyError , e:
				print 'Warning' , value ,'is not be set!'
		result.append('</head>')

		result = '\n'.join(result)
		return result

	def getDivs(self):
		result = []

		for value in ['nav' , 'table']:
			result.append('<div class="container">')
			try:
				result.append( '\n'.join(self[value]) )
			except KeyError , e:
				print 'Warning' , value ,'is not be set!'				
			result.append('</div>')

		result = '\n'.join(result)
		return result


	def getBody(self):
		result = ['<body>',
			self.getDivs(),
			 '</body>'
			]

		result = '\n'.join(result)
		return result

	def display(self):
		result = [
			'<!DOCTYPE html>',
			'<html lang="zh-CN">',
			self.getHeader(),
			self.getBody(),
			'</html>'
			]
		result = '\n'.join(result)
		return result

	def addFile(self , fname):
		try:
			result = ['<table class="table table-striped">']
			result.append('<tbody>')
			f = file(fname , 'r')

			docs = 1
			result.append('<tr class="success">')
			result.append( '<td style="width:60px"><b>' +  u'作文' + str(docs)  + '</b></td>')
			result.append('</tr>')
			docs += 1
			index = 1

			flag = True

			for line in f:
				try:
					value = line.split()[0]
					content = ''.join(line.split()[1:])					
					content = content.decode('utf-8').strip('\n')
				except UnicodeEncodeError , e:
					print e

				if value.startswith('**'):
					result.append('<tr class="success">')
					result.append( '<td style="width:60px"><b>' +  u'作文' + str(docs)  + '</b></td>')
					result.append('</tr>')
					docs += 1
					index = 1
					flag = True
				else:
					if not flag:
						continue
					result.append('<tr>')
					result.append( self.wrapit( 'td' , str(index) ) )
					result.append( self.wrapit( 'td' , value ) )
					result.append( self.wrapit( 'td' , content ) )
					result.append('</tr>')				
					index += 1
					if eval(value) == 0:
						flag = False

			result.append('</tbody>')
			result.append('</table>')
			self.pushit('table' , '\n'.join(result))

			f.close()

		except IOError , e:
			print e

	def removeFile(self):
		if 'table' in self:
    			del self['table']

	def outputHtml(self , fname):
		try:
			f = file(fname , 'w')
			f.write( self.display().encode('utf-8') )
			f.close()
		except IOError , e:
			print e


def main():
	page = HtmlPage(title= u'演示')

	for value in ['zhuangya_sentence' , 'like_sentence' , 'parallel_sentence' , 'textrank_sentence']:
		page.addFile(value + '.txt')
		page.outputHtml(value + '.html')
		page.removeFile()	


if __name__ == "__main__":
	main()