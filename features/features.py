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
import utils.pagerank

class Feature(utils.SaveLoad):

	def __init__(self , datapath = '../data'):
		self.datapath = datapath if datapath.endswith('/') else datapath + '/'
		jieba.setLogLevel('NOTSET')

	def read(self , fname = 'sample_sentence.txt' , redo = False):
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


	def readstopwords(self ,fname='stopwords.txt' , redo = False):
		if getattr(self , 'stopwords' , None) is None or redo:
			f = file(self.datapath + fname ,'r')
			self.stopwords = [ line.strip('\n').decode('utf-8') for line in f ]
			f.close()

		if getattr(self, 'docs_limit' , None) is None or redo:
			self.docs_limit = [ [ [ word for word in sentence if word not in self.stopwords] for sentence in doc] for doc in self.docs_norm]

	def readassociated(self ,fname='associated_words.txt', redo = False):
		if getattr(self , 'associated' , None) is None or redo:
			f = file(self.datapath + fname,'r')
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
			self.feature1_scale = True
			
 
	def solve_feature2(self , redo = False):
		if getattr(self , 'feature2' , None) is None or redo:
			self.feature2 = [ [ len( sentence ) for sentence in doc ] for doc in self.docs_norm ]
			self.feature2_scale = True
			

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
			self.feature4_scale = True

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
			self.feature6 = [ [ len(set(sentence)) for sentence in doc ] for doc in self.docs_limit ]
			self.feature6_scale = True
			


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
			self.feature8 = [ [ len(set( [ word for word in sentence if  len(word) == 4 ] )) for sentence in doc ] for doc in self.docs_limit ]
			self.feature8_scale = True
			

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
			self.feature10_scale = True
			

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
					propertyvec = [0.0] * vecsize
					propertyvec = np.array(propertyvec)

					for word in sentence:
						propertyvec[ propertyindex[ utils.property(word) ] ] += 1.0
					
					propertyvec  = utils.normalize(propertyvec)

					entropy = np.sum (  [ - x * np.log2(x) for x in propertyvec if not x == 0] )

					propertyvec = list(propertyvec)
					propertyvec.append( entropy )

					docvec.append(  propertyvec  )

				veclast = [ vec[-1] for vec in docvec]
				veclast = utils.scalemax(veclast)
				for i in range(len(doc)):
					docvec[i][-1] = veclast[i]

				self.feature12.append( docvec )


	def solve_feature13(self , redo = False , df_min = 10.0):
		if getattr(self , 'feature13' , None) is None or redo:

			self.readstopwords()

			idf = defaultdict(float)
			sentenceTot = 0.0

			for doc in self.docs_limit:
				for sentence in doc:
					sentenceTot += 1.0
					for word in set(sentence):
						idf[word] += 1.0			

			wordkey = sorted( [ (word , idf[word]) for word in idf ] , key = lambda x : -x[1] )
			wordkey = [ word for word , value in wordkey if value >= df_min ]

			for word in idf:
				idf[word] = np.log( sentenceTot / idf[word] )

			self.feature13 = []
			self.feature13_size = len(wordkey)

			for doc in self.docs_limit:
				doctfidf = []
				for sentence in doc:
					vec = [0.0] * self.feature13_size
					for word in sentence:
						if word in wordkey:
							vec[ wordkey.index(word) ] += 1.0
					if len(sentence):
						vec = [ x / len(sentence) * idf[wordkey[index]] for index , x   in enumerate(vec)]

					vec = utils.scalemax(vec)

					doctfidf.append(vec)

				self.feature13.append(doctfidf)

	def solve_feature14(self , redo = False):
		if getattr(self , 'feature14' , None) is None or redo:

			self.readwordlevel()
			self.feature14 = []
			self.feature14_size = 6
			for doc in self.docs_norm:
				docvec = []
				for sentence in doc:
					vec = [0.0] * 5
					for word in sentence:
						try:
							vec[ self.wordlevel[word] ] += 1.0
						except KeyError , e:
							vec[4] += 1.0

					vec = utils.normalize(vec)

					entropy = np.sum (  [ - x * np.log2(x) for x in vec if not x == 0 ]  )

					vec.append(entropy)

					docvec.append(vec)

				veclast = [ vec[-1] for vec in docvec]
				veclast = utils.scalemax(veclast)
				for i in range(len(doc)):
					docvec[i][-1] = veclast[i]

				self.feature14.append(docvec)



	def solve_feature15(self , redo = False):
		if getattr(self , 'feature15' , None) is None or redo:
			self.readembedded()
			self.feature15 = [ [ len( [word for word in sentence if len(word) == 2 and self._startswith(word , self.embedded[0]) ] ) for sentence in doc ] for doc in self.docs_norm ]
			self.feature15_scale = True
			

	def solve_feature16(self , redo = False):
		if getattr(self , 'feature16' , None) is None or redo:
			self.readembedded()
			self.feature16 = [ [ len(set( [word for word in sentence if len(word) == 2 and self._startswith(word , self.embedded[0]) ] ) ) for sentence in doc ] for doc in self.docs_norm ]
			self.feature16_scale = True
			

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

			self.feature17_scale = True
			

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

			self.feature18_scale = True


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
			
			self.feature27_scale = True

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
			
			self.feature28_scale = True

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
					vec = utils.normalize(vec)
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

			self.feature30_scale = True
			
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

				if sub > 3:
					return result

				edis = editdistance( utils.property(segments[index]) , basep ) 
				rate =  edis / max( float(len(segments[index])) ,  basel )

				if rate > 0.3:
					return result

				samenum = 0

				for x in segments[0]:
					if x in segments[index]:
						samenum += 1

				if samenum < 2:
					return result

				if sub == 0:
					result += 1
				else:
					result += 1

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

					docvec.append( 1 if parallel > 0.2 else 0 )

				for index in range(len(doc)):
					docvec[index] += checkparallel( doc[index:] )

				self.feature31.append( docvec )

			self.feature31_scale = True
			


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

			self.feature32_scale = True
			

	def solve_feature33(self , redo = False):
		if getattr(self , 'feature33' , None) is None or redo:
			self.readstopwords()
			# prepare for lda training
			ldadocs = []
			
			for doc in self.docs_limit:
				for sentence in doc:
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
			self.feature33_size = len(self.feature33[0][0])
	
	def solve_feature34(self , redo = False):
		if getattr(self , 'feature34' , None) is None or redo:
			self.readstopwords()
			# prepare for lda training
			ldadocs = []	
			for doc in self.docs_limit:				
				ldadocs.append( ' '.join( [ ' '.join(sentence) for sentence in doc ] ) ) 

			f = file(self.datapath + 'lda_input_100.txt','w')
			f.write( str(len(ldadocs)) + '\n')
			for sentence in ldadocs:
				f.write( sentence.encode('utf-8') + ' ')
				f.write('\n')
			f.close()

			# read data after lda training
			sentencesvec = []
			f = file(self.datapath + 'lda_input.txt.theta','r')
			for index in range(len(self.docs_limit)):
				docvec = []
				for index2 in range(len(self.docs_limit[index])):
					vec = f.readline().split()
					vec = [ eval(x) for x in vec ]
					docvec.append(vec)
				sentencesvec.append(docvec)
			f.close()

			#read doc lda vector
			docsvec = []
			f = file(self.datapath + 'lda_input_100.txt.theta','r')
			for index in range(len(self.docs_limit)):
				vec = f.readline().split()
				vec = [ eval(x) for x in vec ]
				docsvec.append(vec)
			f.close()

			self.feature34 = []
			for index in range(len(sentencesvec)):
				docvec = []
				for index2 in range(len(sentencesvec[index])):
					docvec.append( np.dot(docsvec[index] , sentencesvec[index][index2]) )
				self.feature34.append(docvec)


	def solve_feature35(self , redo = False):
		if getattr(self , 'feature35' , None) is None or redo:
			self.readstopwords()
			# prepare for lda training
			ldadocs = []	
			for doc in self.docs_limit:				
				ldadocs.append( ' '.join( [ ' '.join(sentence) for sentence in doc ] ) ) 

			f = file(self.datapath + 'lda_input_100.txt','w')
			f.write( str(len(ldadocs)) + '\n')
			for sentence in ldadocs:
				f.write( sentence.encode('utf-8') + ' ')
				f.write('\n')
			f.close()

			# read data after lda training
			sentencesvec = []
			f = file(self.datapath + 'lda_input.txt.theta','r')
			for index in range(len(self.docs_limit)):
				docvec = []
				for index2 in range(len(self.docs_limit[index])):
					vec = f.readline().split()
					vec = [ eval(x) for x in vec ]
					docvec.append(vec)
				sentencesvec.append(docvec)
			f.close()

			#read doc lda vector
			docsvec = []
			f = file(self.datapath + 'lda_input_100.txt.theta','r')
			for index in range(len(self.docs_norm)):
				vec = f.readline().split()
				vec = [ eval(x) for x in vec ]
				docsvec.append(vec)
			f.close()

			self.feature35 = []
			for index in range(len(sentencesvec)):
				docvec = []
				maxindex =  sorted( enumerate(docsvec[index]) , key = lambda x : -x[1] )[0][0]
				for index2 in range(len(sentencesvec[index])):
					docvec.append(sentencesvec[index][index2][maxindex])
				self.feature35.append(docvec)
		

	def solve_feature36(self , redo = False):
		if getattr(self , 'feature36' , None) is None or redo:
			self.readstopwords()
			# prepare for word2vec training
			w2vdocs = []
			
			for doc in self.docs_limit:
				for sentence in doc:
					w2vdocs.append( sentence )

			f = file(self.datapath + 'w2v_input.txt','w')
			for sentence in w2vdocs:
				for word in sentence:
					f.write( word.encode('utf-8') + ' ')
				f.write('\n')
			f.close()

			# read data after word2vec training
			self.feature36 = []
			f = file(self.datapath + 'w2v_output.txt','r')
			for doc in self.docs_norm:
				docvec = []
				for sentence in doc:
					vec = f.readline().split()
					vec = [ eval(x) for x in vec ]
					docvec.append(vec)

				self.feature36.append(docvec)
			f.close()
			self.feature36_size = len(self.feature36[0][0])


	def solve_feature37(self , redo = False  , niter = 1000 ,  d = 0.85):
		if getattr(self , 'feature37' , None) is None or redo:
			self.readstopwords()
			
			self.feature37 = []
			for doc in self.docs_limit:
				self.feature37.append(utils.pagerank.solve(doc))		

	def solve_feature38(self , redo = False):
		if getattr(self , 'feature38' , None) is None or redo:
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

			self.feature38 = []

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

				self.feature38.append(docvec)

			self.feature38_scale = True



	def solve_feature39(self , redo = False):
		if getattr(self , 'feature39' , None) is None or redo:
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

			self.feature39 = []

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

				self.feature39.append(docvec)

			self.feature39_scale = True
			

	def solve_feature40(self , redo = False):
		if getattr(self, 'feature40' , None) is None or redo:

			self.feature40 = []
			for doc in self.docs_norm:
				docvec = []
				for index , sentence in enumerate(doc):
					docvec.append(index)
				self.feature40.append(docvec)
			
			self.feature40_scale = True


	def solve_feature41(self , redo = False):
		if getattr(self, 'feature41' , None) is None or redo:

			self.feature41 = []
			for doc in self.docs_norm:
				docvec = []
				for index , sentence in enumerate(doc):
					docvec.append( len(doc) - index - 1)
				self.feature41.append(docvec)
			
			self.feature41_scale = True


	def solve_all(self ,  full = True , featurelist = []):
		if full:
			featurelist = [ 'feature' + str(i) for i in range(1 , 42) ]
		else:
			featurelist = [ 'feature' + str(i) for i in featurelist ]

		for attrname in featurelist:
			if getattr(self, 'solve_' + attrname , None) is not None:
				getattr( self , 'solve_' + attrname )()

	def getvec(self , full = True , decfeaturenums = []):
		if full:
			featurelist = [ 'feature' + str(i) for i in range(1 , 42) ]
		else:
			featurelist = [ 'feature' + str(i) for i in range(1 , 42)  if  i not in decfeaturenums]

		result = []
		for index in range(len(self.docs)):
			docvec = []
			for index2 in range(len(self.docs[index])):
				docvec.append( [] )
			result.append(docvec)

		for attrname in featurelist:
			if getattr(self, attrname , None) is not None:
				if getattr(self, attrname + '_size' , None) is not None:
					for index in range(len( getattr( self , attrname ) ) ):
						for index2 in range(len( getattr( self , attrname )[index] )):
							result[index][index2] += getattr( self , attrname )[index][index2]
				elif getattr(self, attrname + '_scale' , None) is None:
					for index in range(len( getattr( self , attrname ) ) ):
						for index2 in range(len( getattr( self , attrname )[index] )):
							result[index][index2].append( getattr( self , attrname )[index][index2] )
				else:
					for index in range(len( getattr( self , attrname ) ) ):
						docvec = getattr( self , attrname )[index]
						docvec = utils.scalemax(docvec)
						for index2 in range(len( docvec )):
							result[index][index2].append( docvec[index2] )

		return result