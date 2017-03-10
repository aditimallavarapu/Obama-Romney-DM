clean the data	-- python
write clean data into a file	--python
fix missing values	--manual
	fix time ---- lot of missing values 
	for now deleting the data and time columns
break into n grams	now only unigram
find frequencies	python



Drawbacks:

If i remove date and time, we need to remove date and time from the test file too-- java script????
 
 
Next Steps:
write frequencies to a file????
N- fold how?
may be try it in weka after having written to the file
calculate recall precision fscore for current training
integrate the other training sheet also - Romney


Results: 

1) 6000 features Obama
Class 1
('Precision ', 0.4883720930232558)
('Recall ', 0.6098484848484849)
('F1Score ', 0.5423919146546884)
Class 0
('Precision ', 0.4734576757532281)
('Recall ', 0.40441176470588236)
('F1Score ', 0.43621943159286186)
Class -1
('Precision ', 0.5266106442577031)
('Recall ', 0.47534766118836913)
('F1Score ', 0.4996677740863788)
('Overall Accuracy ', 0.49541666666666667)
Model built...

2) 6000 features Romney
Class 1
('Precision ', 0.5030364372469636)
('Recall ', 0.6046228710462287)
('F1Score ', 0.549171270718232)
Class 0
('Precision ', 0.4739010989010989)
('Recall ', 0.4328732747804266)
('F1Score ', 0.4524590163934426)
Class -1
('Precision ', 0.5394736842105263)
('Recall ', 0.47307692307692306)
('F1Score ', 0.5040983606557378)
('Overall Test Accuracy ', 0.5045833333333334)
Model built...
 