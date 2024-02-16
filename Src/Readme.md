
----------------------
# Baseline.
Unihar: 
	1. Self-supervised learning for *feature extraction* with massive unlabeled data for *all users*.  [Using complete aug.]
	2. *Limited labeled data* from the source users.  [Using complete+approximate]

----------------------
# Our Scheme. 
Phar: 
	1. Measure domain distance. {Better than directly pre-train on all users.}  	--New add features 1; 
	-------- 
	2. Remove domain distance.  													--New add features 2; 
	3. Refine pseudo-label. {Former do!} 

Kernel points:
	1. Through Beatrix+gram.py
	2. Through grl+nwd.py
	3. through cotta.py

Appendix: 
	match_utils.py little performance. 

Start point:  
	0. argPars.py model.py (model architectures.)
	1. pretrain.py
	2. cotta.py
Dataload:  
	1. datasetPre.py	dataAug.py	

