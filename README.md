# Beautiful sentence
*A set of code for finding beautiful sentences in the Chinese article*

The project is used for some personal experiments

These algorithms are cited by a lot of papers by Chinese researchs and foreign researchs.



#API

#####FIRST STEP
The sample is "prepare.py" create a class instance for use it

    obj = features.Feature( datapath )
    obj.save("features/featureobj")
    
#####SECOND STEP
The all data for reading should be under datapath , program reads the data by the filename otherwise read the default file

    obj.read(fname = "sample_sentence.txt" )   #the articles after cutting , split with "********************"
    obj.readorigin(fname = "sample_merge.txt")  #the articles before cutting , split with "********************"
    obj.readstopwords(fname="stopwords.txt" )
    
Some other data such as "associated_words" , "word_level" ,  i suggest you to take the default data by copying them

#####THIRD STEP

    obj.solve_all(full = True , featurelist = [])
samples:

    obj.solve_all()  -- solve the all features
	obj.solve_all(full=False , featurelist = [1,2,3,4])   #  only solve the 1,2,3,4 features
    obj.getvec(full = True , decfeaturenums = [])
samples:

	obj.getvec()  # get the all features
	obj.getvec(full=False , decfeaturenums = [1,2,3,4])   #  only get the range(5,42) features


I'm sorry about some features like 'lda' , 'word2vec' which needs other program.

The data will to be write to the file 'lda_input.txt' , 'w2v_input.txt' .

please run it by yourself

Otherwise you can ignore the feature (33,34,35,36) by use 

    obj.solve_all(full=False , featurelist = range(1,33) + range(37,42))
    obj.getvec(full=False , decfeaturenums = [33,34,35,36])
Thes feature methods may be updated in furture

#####FOURTH STEP
The sample is "cross_validation_svm.py"

The featureobj will be saved by the "prepare.py"

    obj.save("features/featureobj")
    
And then you can read it and do the cross validation 
the result will be written to the svm / svmrank dir 



#License

**The project is open source project released under the [GNU GPLv2 license](http://www.gnu.org/licenses/gpl-2.0.html) Copyright (c) 2016-now**
