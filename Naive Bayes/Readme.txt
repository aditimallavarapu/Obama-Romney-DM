************************Readme:**********************

Naive bayes:
1) Run the build model : calls Trainmodel and Preprocess classes
	Take the excel file
	converts into two tab separated files
	cleans the tweets and writes in files
	Reads those files and constructs features 
	Constructs models from unigram, bigram, trigram, quadgram and stores in pickle files
	
2) Run testmodel:
		takes the test excel file
		converts into two tab separated files
		cleans tweets writes into another file
		reads those files and the model constructed in step 1 
		applies test features 
		reports metrics
		
Reported metrics:

Starting... Unigram Obama
accuracy:  0.517435897436
Class 1
('Precision ', 0.3664036076662909)
('Recall ', 0.5584192439862543)
('F1Score ', 0.4424778761061947)
Class 0
('Precision ', 0.40425531914893614)
('Recall ', 0.4185022026431718)
('F1Score ', 0.41125541125541126)
Class -1
('Precision ', 0.4938118811881188)
('Recall ', 0.5807860262008734)
('F1Score ', 0.5337792642140469)
('Overall Test Accuracy ', 0.42041666666666666)
Starting... Bigram Obama
accuracy: 0.474312402698
Class 1
('Precision ', 0.36743515850144093)
('Recall ', 0.4473684210526316)
('F1Score ', 0.40348101265822783)
Class 0
('Precision ', 0.3744075829383886)
('Recall ', 0.3516320474777448)
('F1Score ', 0.3626625860749808)
Class -1
('Precision ', 0.40190476190476193)
('Recall ', 0.6178623718887262)
('F1Score ', 0.4870167339873052)
('Overall Test Accuracy ', 0.3845183003786285)
Starting... Trigram Obama
accuracy: 0.406431207169
Class 1
('Precision ', 0.34841628959276016)
('Recall ', 0.14102564102564102)
('F1Score ', 0.20078226857887874)
Class 0
('Precision ', 0.3880597014925373)
('Recall ', 0.07784431137724551)
('F1Score ', 0.12967581047381546)
Class -1
('Precision ', 0.32326283987915405)
('Recall ', 0.9399707174231332)
('F1Score ', 0.4810790558261521)
('Overall Test Accuracy ', 0.32934643314822726)
Starting... quadigram Obama
accuracy 0.379857690203
Class 1
('Precision ', 0.30952380952380953)
('Recall ', 0.025490196078431372)
('F1Score ', 0.04710144927536231)
Class 0
('Precision ', 0.4888888888888889)
('Recall ', 0.03374233128834356)
('F1Score ', 0.06312769010043041)
Class -1
('Precision ', 0.30187814933577645)
('Recall ', 0.9909774436090225)
('F1Score ', 0.4627808988764045)
('Overall Test Accuracy ', 0.30572687224669604)
Starting... Unigram romney
accuracy:  0.548183254344
Class 1
('Precision ', 0.21864594894561598)
('Recall ', 0.5130208333333334)
('F1Score ', 0.3066147859922179)
Class 0
('Precision ', 0.4188235294117647)
('Recall ', 0.3207207207207207)
('F1Score ', 0.363265306122449)
Class -1
('Precision ', 0.6115702479338843)
('Recall ', 0.69375)
('F1Score ', 0.6500732064421669)
('Overall Test Accuracy ', 0.43105590062111804)
Starting... Bigram Romney
accuracy: 0.532275132275
Class 1
('Precision ', 0.21201413427561838)
('Recall ', 0.16216216216216217)
('F1Score ', 0.18376722817764166)
Class 0
('Precision ', 0.5)
('Recall ', 0.07481751824817519)
('F1Score ', 0.13015873015873017)
Class -1
('Precision ', 0.436127744510978)
('Recall ', 0.9287991498405951)
('F1Score ', 0.5935483870967742)
('Overall Test Accuracy ', 0.4115660616293795)
Starting... quadigram romney
accuracy 0.505902192243
Class 1
('Precision ', 0.0)
('Recall ', 0.0)
('F1Score ', 'NAN')
Class 0
('Precision ', 0.10714285714285714)
('Recall ', 0.005649717514124294)
('F1Score ', 0.01073345259391771)
Class -1
('Precision ', 0.4011627906976744)
('Recall ', 0.9889746416758545)
('F1Score ', 0.5707922367165129)
('Overall Test Accuracy ', 0.39421813403416556)		