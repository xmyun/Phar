
---------------------- 
# Baseline. 
Unihar: 
	1. Self-supervised learning for *feature extraction* with massive unlabeled data for *all users*.  [Using complete aug.]
	2. *Limited labeled data* from the source users.  [Using complete+approximate]
非环境错误，是算法上的差异; 
	Total flow: 
		0. The former version: Rename new branch;
		1. Checked the pretrain. (Flowing: argParse, datasetPre, model)
			load_Uci_users.  load_Across_users

----------------------
# Our Scheme. 
Phar: 
	1. Measure domain distance. {Better than directly pre-train on all users.}  	-- New add features 1; 
	-------- 
	2. Remove domain distance.  													-- New add features 2; 
	3. Refine pseudo-label. {Former do!} 

Kernel points:
	1. Through Beatrix+gram.py
	2. Through grl+nwd.py
	3. Through cotta.py 

Appendix: 
	match_utils.py little performance. 

Start point:  
	0. argPars.py model.py (model architectures.)
	1. pretrain.py
	2. cotta.py

Dataload:  
	1. datasetPre.py	dataAug.py	


---------------------
Dataset loading; 
Measure distance, 
Large, 
Small. 
The main thoughts. Going forwards. 



