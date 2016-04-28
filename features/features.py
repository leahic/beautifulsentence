# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import sys
import time
from collections import defaultdict
import jieba
import jieba.posseg as pseg
import gensim
import numpy as np
import utils

class Feature(utils.SaveLoad):

	def __init__(self , datapath = '../data'):
		self.datapath = datapath if datapath.endswith('/') else datapath + '/'
		jieba.setLogLevel('NOTSET')

	def read(self , fname , redo = False):
		if getattr(self ,  'docs' , None) is None or redo:
			f = file(self.datapath + fname , 'r')
			self.docs = []
			doc = []
			for line in f:
				if line.startswith('********************'):
					self.docs.append(doc)
					doc = []
				else:
					doc.append( line.strip('\n').decode('utf-8') )  	
			f.close()

		if getattr(self , 'docs_norm' , None) is None or redo:
			punctuation = [u'，' , u'。' , u'…'  , u'；' , u'：' , u'“' ,u'”', u'、' ,  u'+' , u'-' , u'！' , u'—' , u'《' , u'》' , u'？' , u'（' , u'）'  ,u'·' , u'!' ,u'’' ,u'‘' ,u'.' ,u',']
			punctuation.append('')		
			self.docs_norm = [ [ [ word for word in sentence.split(' ') if word not in punctuation ] for sentence in doc ] for doc in self.docs ]			

	def readorigin(self , fname = 'sample_merge.txt' , redo = False):
		if getattr(self , 'origin' , None) is None or redo:
			f = file(self.datapath + fname , 'r')
			self.origin = []
			doc = []
			for line in f:
				if line.startswith('********************'):
					self.origin.append(doc)
					doc = []
				else:
					doc.append( line.strip('\n').decode('utf-8') )
			f.close()


	def readstopwords(self , redo = False):
		if getattr(self , 'stopwords' , None) is None or redo:
			f = file(self.datapath + 'stopwords.txt','r')
			self.stopwords = [ line.strip('\n').decode('utf-8') for line in f ]
			f.close()

	def readassociated(self , redo = False):
		if getattr(self , 'associated' , None) is None or redo:
			f = file(self.datapath + 'associated_words.txt','r')
			self.associated = [ line.strip('\n').decode('utf-8') for line in f ]
			f.close()		

	def readwordlevel(self , redo = False):
		if getattr(self , 'wordlevel' , None) is None or redo:
			self.wordlevel = dict()
			for index , target  in enumerate( ['D' , 'C' , 'B' , 'A'] ):
				f = file(self.datapath + 'word_level/word_level_' + target + '.txt', 'r')
				for line in f:
					self.wordlevel[ line.strip('\n').decode('utf-8') ]  = 3 - index
				f.close()

	def readembedded(self , redo = False):
		if getattr(self , 'embedded' , None) is None or redo:
			self.embedded = []
			f = file(self.datapath + 'embedded_word_two.txt','r')
			self.embedded.append( [ line.strip('\n').decode('utf-8') for line in f ] )
			f.close()

			f = file(self.datapath + 'embedded_word_four.txt','r')
			self.embedded.append( [ line.strip('\n').decode('utf-8') for line in f ] )
			f.close()

	def _startswith(self , target , wordlist):
		for word in wordlist:
			if target.startswith(word):
				return True
		return False

	def solve_feature1(self , redo = False):
		if getattr(self , 'feature1' , None) is None or redo:
			self.feature1 = [ [ len( ''.join(sentence) ) for sentence in doc ] for doc in self.docs_norm ]
 
	def solve_feature2(self , redo = False):
		if getattr(self , 'feature2' , None) is None or redo:
			self.feature2 = [ [ len( sentence ) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature3(self , redo = False):
		if getattr(self , 'feature3' , None) is None or redo:
			self.solve_feature1()
			self.solve_feature2()
			np.seterr(all = 'ignore')
			self.feature3 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature1[index] , dtype = np.float32)
				y  =  np.array( self.feature2[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature3.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature4(self , redo = False):
		if getattr(self , 'feature4' , None) is None or redo:
			self.feature4 = [ [ len(set(sentence)) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature5(self , redo = False):
		if getattr(self , 'feature5' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature4()
			np.seterr(all = 'ignore')
			self.feature5 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature4[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature5.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature6(self , redo = False):
		if getattr(self , 'feature6' , None) is None or redo:
			self.readstopwords()
			self.feature6 = [ [ len(set( [ word for word in sentence if word not in self.stopwords] )) for sentence in doc ] for doc in self.docs_norm ]


	def solve_feature7(self , redo = False):
		if getattr(self , 'feature7' , None) is None or redo:
			self.solve_feature4()
			self.solve_feature6()
			np.seterr(all = 'ignore')
			self.feature7 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature4[index] , dtype = np.float32)
				y  =  np.array( self.feature6[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature7.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature8(self , redo = False):
		if getattr(self , 'feature8' , None) is None or redo:
			self.readstopwords()
			self.feature8 = [ [ len(set( [ word for word in sentence if word not in self.stopwords and len(word) == 4 ] )) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature9(self , redo = False):
		if getattr(self , 'feature9' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature8()
			np.seterr(all = 'ignore')
			self.feature9 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature8[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature9.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature10(self , redo = False):
		if getattr(self , 'feature10' , None) is None or redo:
			self.readassociated()
			self.feature10 = [ [ len( [ word for word in sentence if word in self.associated ] ) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature11(self , redo = False):
		if getattr(self , 'feature11' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature10()
			np.seterr(all = 'ignore')
			self.feature11 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature10[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature11.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature12(self , redo = False , vecsize = 10):
		if getattr(self , 'feature12' , None) is None or redo:
			if vecsize < 2:
				raise('vecsize is less than 2')

			vecsize -= 1

			propertydict = defaultdict(int)

			for doc in self.docs_norm:
				for sentence in doc:
					for word in sentence:
						propertydict[ utils.property(word) ] += 1

			propertydict = sorted( [  (word , propertydict[word])  for word in propertydict ] , key = lambda x : - x[1] )

			propertyindex = defaultdict(int)

			vecsize -= 1

			for index , (word , value) in enumerate(propertydict):
				propertyindex[word] = index if index < vecsize else vecsize

			vecsize += 1

			self.feature12 = []
			self.feature12_size = vecsize + 1

			for doc in self.docs_norm:

				docvec = []

				for sentence in doc:
					length = float( len(sentence) )
					propertyvec = [0.0] * vecsize
					propertyvec = np.array(propertyvec)

					for word in sentence:
						propertyvec[ propertyindex[ utils.property(word) ] ] += 1.0
					
					propertyvec  = propertyvec / length

					entropy = np.sum (  [ - x * np.log2(x) for x in propertyvec if not x == 0] )

					propertyvec = list(propertyvec)
					propertyvec.append( entropy )

					docvec.append(  propertyvec  )

				self.feature12.append( docvec )

	def solve_feature13(self , redo = False , df_min = 10.0):
		if getattr(self , 'feature13' , None) is None or redo:

			self.readstopwords()

			idf = defaultdict(float)
			sentenceTot = 0.0

			for doc in self.docs_norm:
				for sentence in doc:
					sentenceTot += 1.0
					for word in set(sentence):
						idf[word] += 1.0			

			wordkey = sorted( [ (word , idf[word]) for word in idf  if word not in self.stopwords ] , key = lambda x : -x[1] )
			wordkey = [ word for word , value in wordkey if value >= df_min ]

			for word in idf:
				idf[word] = np.log( sentenceTot / idf[word] )

			self.feature13 = []
			self.feature13_size = len(wordkey)

			for doc in self.docs_norm:
				doctfidf = []
				for sentence in doc:
					vec = [0.0] * self.feature13_size
					for word in sentence:
						if word in wordkey:
							vec[ wordkey.index(word) ] += 1.0
					if len(sentence):
						vec = [ x / len(sentence) * idf[wordkey[index]] for index , x   in enumerate(vec)]

					doctfidf.append(vec)

				self.feature13.append(doctfidf)
	def solve_feature14(self , redo = False):
		if getattr(self , 'feature14' , None) is None or redo:

			self.readwordlevel()
			self.feature14 = []

			for doc in self.docs_norm:
				docvec = []
				for sentence in doc:
					vec = [0.0] * 5
					for word in sentence:
						try:
							vec[ self.wordlevel[word] ] += 1.0
						except KeyError , e:
							vec[4] += 1.0

					if len(sentence):					
						vec = list( np.array( vec ) / len(sentence) )

					entropy = np.sum (  [ - x * np.log2(x) for x in vec if not x == 0 ]  )

					vec.append(entropy)

					docvec.append(vec)

				self.feature14.append(docvec)


	def solve_feature15(self , redo = False):
		if getattr(self , 'feature15' , None) is None or redo:
			self.readembedded()
			self.feature15 = [ [ len( [word for word in sentence if len(word) == 2 and self._startswith(word , self.embedded[0]) ] ) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature16(self , redo = False):
		if getattr(self , 'feature16' , None) is None or redo:
			self.readembedded()
			self.feature16 = [ [ len(set( [word for word in sentence if len(word) == 2 and self._startswith(word , self.embedded[0]) ] ) ) for sentence in doc ] for doc in self.docs_norm ]

	def solve_feature17(self , redo = False):
		if getattr(self , 'feature17' , None) is None or redo:
			self.readembedded()
			self.feature17 = []
			for doc in self.docs:
				docvec = []
				for sentence in doc:
					sentence = sentence.split(' ')
					wordnums = 0
					length = len(sentence)
					for index , word in enumerate(sentence):
						if len(word) == 2 and index < length - 1 and len(sentence[index + 1]) == 2 and self._startswith(word , self.embedded[1]):
							wordnums += 1
						elif len(word) == 4 and self._startswith(word , self.embedded[1]):
							wordnums += 1

					docvec.append( wordnums )

				self.feature17.append( docvec )

	def solve_feature18(self , redo = False):
		if getattr(self , 'feature18' , None) is None or redo:
			self.readembedded()
			self.feature18 = []
			for doc in self.docs:
				docvec = []
				for sentence in doc:
					sentence = sentence.split(' ')
					wordtarget = set()
					length = len(sentence)
					for index , word in enumerate(sentence):
						if len(word) == 2 and index < length - 1 and len(sentence[index + 1]) == 2 and self._startswith(word , self.embedded[1]):
							wordtarget.add(word + sentence[index + 1])
						elif len(word) == 4 and self._startswith(word , self.embedded[1]):
							wordtarget.add(word)

					docvec.append( len(wordtarget) )

				self.feature18.append( docvec )


	def solve_feature21(self , redo = False):
		if getattr(self , 'feature21' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature15()
			np.seterr(all = 'ignore')
			self.feature21 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature15[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature21.append( [ x if not np.isnan(x) else 0.0 for x in rate ])	

	def solve_feature22(self , redo = False):
		if getattr(self , 'feature22' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature16()
			np.seterr(all = 'ignore')
			self.feature22 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature16[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature22.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature23(self , redo = False):
		if getattr(self , 'feature23' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature17()
			np.seterr(all = 'ignore')
			self.feature23 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature17[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature23.append( [ x if not np.isnan(x) else 0.0 for x in rate ])

	def solve_feature24(self , redo = False):
		if getattr(self , 'feature24' , None) is None or redo:
			self.solve_feature2()
			self.solve_feature18()
			np.seterr(all = 'ignore')
			self.feature24 = []
			for index in range( len(self.docs_norm) ):
				x  =  np.array( self.feature2[index] , dtype = np.float32)
				y  =  np.array( self.feature18[index] , dtype = np.float32)
				rate = list(y / x)
				self.feature24.append( [ x if not np.isnan(x) else 0.0 for x in rate ])


	def solve_feature27(self , redo = False):
		if getattr(self , 'feature27' , None) is None or redo:
			partdot = [ u'，' , u'。' , u'；' , u'：' , u',' , u'！' , u'？' , u'!' , u'… …' ] 

			self.feature27 = []
			for doc in self.docs:
				docvec = []
				for sentence in doc:
					sentence = [ sentence.replace(' ' , '') ]
					for dot in partdot:
						nexturn = []
						for part in sentence:
							for seg in part.split(dot):
								nexturn.append(seg)
						sentence = nexturn

					docvec.append( len( [ part for part in sentence if len(part) > 0 ]) )

				self.feature27.append( docvec )


	def solve_feature28(self , redo = False):
		if getattr(self , 'feature28' , None) is None or redo:

			partdot = [ u'，' , u'。' , u'；' , u'：' , u',' , u'！' , u'？' , u'!' , u'… …' ] 

			self.feature28 = []
			for doc in self.docs:
				docvec = []
				for sentence in doc:
					sentence = [ sentence.replace(' ' , '') ]
					for dot in partdot:
						nexturn = []
						for part in sentence:
							for seg in part.split(dot):
								nexturn.append(seg)
						sentence = nexturn
					sentence = [ part for part in sentence if len(part) > 0 ]
					average = 0.0
					for part in sentence:
						average += len(part)
					if len(sentence) > 0:
						average /= len(sentence)
					docvec.append( average )

				self.feature28.append( docvec )

	def solve_feature29(self , redo = False):
		if getattr(self , 'feature29' , None) is None or redo:

			punctuation = [
					[u'，' , u',' ],
					[u'；' , u';'],
					[u'：' , u':'],
					[u'“' , u'”' ,u'’' ,u'‘' ,  u"'" , u'"'],
					[ u'（' , u'）' , u'(' , u')'],
					[u'、'],
					[u'—' , u'-'],
					[u'…' , u'...'],
					[u'。' , u'.'],
					[u'！',u'!'],
					[u'？' , u'?']
				           ]

			self.feature29 = []
			self.feature29_size = len(punctuation)

			for doc in self.docs:
				docvec = []
				for sentence in doc:
					vec = [0] * self.feature29_size
					for w in sentence:
						for index , flags in enumerate(punctuation):
							if w in flags:
								vec[index] += 1
					docvec.append(vec)

				self.feature29.append(docvec)


	def solve_feature30(self , redo = False):
		if getattr(self , 'feature30' , None) is None or redo:

			nondecorate = ['n' , 's' , 'o' , 't' , 'v' , 'x']
			decorate = ['a' , 'b' , 'd' , 'm' , 'q' , 'f']

			analogyword = [u'像',u'好像',u'像是',u'似',u'好似',u'恰似',u'似的',u'若',u'如同',u'仿佛',u'犹如',u'有如',u'好比']

			self.feature30 = []
			
			for doc in self.docs_norm:
				docvec = []
				for sentence in doc:
					analogynum = 0
					flag = -1
					for index , word in enumerate(sentence):					
						if  flag != -1 and self._startswith( utils.property(word) , ['n'] ):
							propertymiddle = utils.property( ''.join( sentence[flag + 1:index] ) )
							choose = 0
							if len(propertymiddle) == 0:
								choose = 1
							for x in propertymiddle:
								if x in nondecorate:
									choose = -1
									break
								if x in decorate:
									choose = 1
							if choose > 0:
								analogynum += 1
							flag = -1
						elif word in analogyword:
							flag = index

					docvec.append(analogynum)

				self.feature30.append(docvec)


	def solve_feature31(self , redo = False):

		def editdistance(s1 , s2):
			l1 = len(s1) + 1
			l2 = len(s2) + 1
			f = []
			for i in range(l1):
				f.append( [0] * l2 )

			for i in range(l1):
				f[i][0] = i
			for j in range(l2):
				f[0][j] = j

			for i in range(1 , l1):
				for j in range(1 , l2):
					f[i][j] = min( f[i - 1][j] + 1 , f[i][j - 1] + 1 , f[i - 1][j - 1] + ( 1 if s1[i - 1] != s2[j - 1] else 0 ) )

			return f[l1 - 1][l2 - 1]		


		def checkparallel(segments):

			result = 0.0

			basep = utils.property(segments[0])
			basel = float( len(segments[0]) )

			for index in range( 1 , len(segments) ):
				sub =  abs( len(segments[index]) - basel )

				if sub > 7:
					return result

				edis = editdistance( utils.property(segments[index]) , basep ) 
				rate =  edis / max( float(len(segments[index])) ,  basel )

				if rate > 0.5:
					return result

				samenum = 0

				for x in segments[0]:
					if x in segments[index]:
						samenum += 1

				if samenum < 3:
					return result

				if sub == 0:
					result += 0.1 + 0.25 * (1 - rate) + 0.65 * np.log2(samenum)
				else:
					result += 0.1 / sub + 0.25 *  (1 - rate) + 0.65 * np.log2(samenum)

			return result 
					


		if getattr(self , 'feature31' , None) is None or redo:

			partdot = [ u'，' , u'。' , u'；' , u'：' , u',' , u'！' , u'？' , u'!' , u'… …' ] 

			self.feature31 = []

			for doc in self.docs:
				docvec = []
				for sentence in doc:

					parallel = 0.0

					sentence = [ sentence.replace(' ','') ]
					for dot in partdot:
						nexturn = []
						for part in sentence:
							for seg in part.split(dot):
								nexturn.append(seg)
						sentence = nexturn

					sentence = [ part for part in sentence if len(part) > 0 ]

					for seglen in range( 1 , len(sentence) ):
						for index in range( len(sentence) - seglen  + 1):
							segments = []
							tmp = sentence[index:]
							while tmp != []:
								segments.append( ''.join( tmp[0:seglen]) )
								tmp = tmp[seglen:]								


							parallel += checkparallel( segments )

					average = 0.0
					for seg in sentence:
						average += len(seg)
					if len(sentence):
						average /= len(seg)
						parallel /= average

					docvec.append( parallel )

				self.feature31.append( docvec )


	def solve_feature32(self , redo = False):
		if getattr(self , 'feature32' , None) is None or redo:

			partdot = [ u'，' , u'。' , u'；' , u'：' , u',' , u'！' , u'？' , u'!' , u'… …' ] 

			punctuation = [u'？' , u'?']

			questionword = [u'难道' , u'怎能' , u'怎么能'  , u'怎么可能', u'何曾' , u'不是' , u'真的' , u'又会' , u'何必' ]

			self.feature32 = []
			
			for doc in self.docs:
				docvec = []
				for sentence in doc:

					questionum = 0

					sentence = [ sentence.replace(' ','') ]

					questionflag = []

					length = 0

					for x in sentence[0]:
						if x not in partdot:
							length += 1
						elif x in punctuation:
							questionflag.append(length)

					for dot in partdot:
						nexturn = []
						for part in sentence:
							for seg in part.split(dot):
								nexturn.append(seg)
						sentence = nexturn

					sentence = [ part for part in sentence if len(part) > 0 ]

					length = 0

					for seg in sentence:
						length += len(seg)
						if length in questionflag:
							founded = False
							for word in questionword:
								if seg.find(word) != -1:
									founded = True
									break
							if founded:
								questionum += 1

					docvec.append(questionum)

				self.feature32.append(docvec)

	def solve_feature33(self , redo = False):
		if getattr(self , 'feature33' , None) is None or redo:
			self.readstopwords()
			# prepare for lda training
			ldadocs = []
			
			for doc in self.docs_norm:
				for sentence in doc:
					sentence = [word for word in sentence if word not in self.stopwords]
					ldadocs.append( sentence )

			f = file(self.datapath + 'lda_input.txt','w')
			f.write( str(len(ldadocs)) + '\n')
			for sentence in ldadocs:
				for word in sentence:
					f.write( word.encode('utf-8') + ' ')
				f.write('\n')
			f.close()

			# read data after lda training
			self.feature33 = []
			f = file(self.datapath + 'lda_input.txt.theta','r')
			for doc in self.docs_norm:
				docvec = []
				for sentence in doc:
					vec = f.readline().split()
					vec = [ eval(x) for x in vec ]
					docvec.append(vec)

				self.feature33.append(docvec)
			f.close()
	
	def solve_feature35(self , redo = False):
		if getattr(self , 'feature35' , None) is None or redo:
			pass

	def solve_feature36(self , redo = False):
		if getattr(self , 'feature36' , None) is None or redo:
			self.readorigin()

			sentencecut = [u'。',u'！',u'？',u'……']

			docparagraph = []

			for doc in self.origin:
				length = 0
				paragraph = []
				for line in doc:
					length += len(line)
					paragraph.append( length )

				docparagraph.append(paragraph)

			self.feature36 = []

			for index , doc in enumerate( self.docs ):

				paranum = 0
				length = 0
				docvec = []
				for sentence in doc:					
					docvec.append(paranum)

					sentence = sentence.replace(' ' , '')
					length += len(sentence)

					if length in docparagraph[index]:
						paranum += 1

				self.feature36.append(docvec)



	def solve_feature37(self , redo = False):
		if getattr(self , 'feature37' , None) is None or redo:
			self.readorigin()

			sentencecut = [u'。',u'！',u'？',u'……']

			docparagraph = []

			for doc in self.origin:
				length = 0
				paragraph = []
				for line in doc:
					length += len(line)
					paragraph.append( length )

				docparagraph.append(paragraph)

			self.feature37 = []

			for index , doc in enumerate( self.docs ):

				paranum = 0
				length = 0
				docvec = []
				for sentence in doc:					
					docvec.append(len(docparagraph[index]) - paranum - 1)

					sentence = sentence.replace(' ' , '')
					length += len(sentence)

					if length in docparagraph[index]:
						paranum += 1

				self.feature37.append(docvec)

	def solve_feature38(self , redo = False):
		if getattr(self, 'feature38' , None) is None or redo:

			self.feature38 = []
			for doc in self.docs_norm:
				docvec = []
				for index , sentence in enumerate(doc):
					docvec.append(index)
				self.feature38.append(docvec)



	def solve_feature39(self , redo = False):
		if getattr(self, 'feature39' , None) is None or redo:

			self.feature39 = []
			for doc in self.docs_norm:
				docvec = []
				for index , sentence in enumerate(doc):
					docvec.append( len(doc) - index - 1)
				self.feature39.append(docvec)