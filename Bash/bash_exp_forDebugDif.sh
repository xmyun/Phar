#!/bin/bash

# timestamp=$(date +%Y%m%d%H%M%S) 
# CroTyp="MCroDom"   
# GPUNum="1"  # Add feature extractor for shoabi+uci. 
# modelP="gru_v3" 
# Feature_extractor_path="extractor/limu_bert_td4" 
# SrouceDomain="aet0" 
# TargetDomain="shoaib4"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
# nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

timestamp=$(date +%Y%m%d%H%M%S) 
GPUNum="5" 
CroTyp="SCroDom" # The cross dataset manner.  
modelP="gru_v3" # v3_limu; v1_unihar.  
# limu_bert_hhar limu_bert_uci limu_bert_motion limu_bert_shoaib limu_bert_usc 
Feature_extractor_path="extractor/limu_bert_motion" # Only for source.  
SrouceDomain="motion3"  
TargetDomain="hhar1"  #'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
PretrainLogFileName="saved/log_0817/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
AdaptLogFileName="saved/log_0817/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
## nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &
