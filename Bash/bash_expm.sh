#!/bin/bash

# # Multiple source, single target domain. 
# # Target domain 1, source domain: all expect taregt ; 
# timestamp=$(date +%Y%m%d%H%M%S)  
# CroTyp="MCroDom"  
# GPUNum="0"  
# Feature_extractor_path="extractor/limu_bert_td3" # unihar_bert_td4 
# modelP="gru_v3" 
# SrouceDomain="aet0" 
# TargetDomain="motion3"  #'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path  --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &
# nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &
# # Target domain uci2, source domain: all expect taregt + (5,6); 
timestamp=$(date +%Y%m%d%H%M%S) 
CroTyp="MCroDom"   
GPUNum="0"  # Add feature extractor for shoabi+uci. 
modelP="gru_v3" 
Feature_extractor_path="extractor/limu_bert_td3" 
SrouceDomain="aet0"  
TargetDomain="motion3"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &


# # Target domain uci2, source domain: all expect taregt + (5,6); 
timestamp=$(date +%Y%m%d%H%M%S) 
CroTyp="MCroDom"   
GPUNum="0"  # Add feature extractor for shoabi+uci. 
modelP="gru_v3" 
Feature_extractor_path="extractor/limu_bert_td2" 
SrouceDomain="aet0"  
TargetDomain="uci2"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &


timestamp=$(date +%Y%m%d%H%M%S) 
CroTyp="MCroDom"   
GPUNum="0"  # Add feature extractor for shoabi+uci. 
modelP="gru_v3" 
Feature_extractor_path="extractor/limu_bert_td1" 
SrouceDomain="aet0"  
TargetDomain="hhar1"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &


timestamp=$(date +%Y%m%d%H%M%S) 
CroTyp="MCroDom"   
GPUNum="0"  # Add feature extractor for shoabi+uci. 
modelP="gru_v3" 
Feature_extractor_path="extractor/limu_bert_td4" 
SrouceDomain="aet0"  
TargetDomain="shoaib4"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &
# # # Target domain motion3, source domain: all expect taregt + (5,6); 
# timestamp=$(date +%Y%m%d%H%M%S)  
# CroTyp="MCroDom"  
# GPUNum="3"  
# SrouceDomain="aet0"  
# TargetDomain="motion3"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &


# # Target domain shoaib4, source domain: all expect taregt + (5,6); 
# timestamp=$(date +%Y%m%d%H%M%S) 
# CroTyp="MCroDom"  
# GPUNum="2"  
# SrouceDomain="aet0" 
# TargetDomain="shoaib4"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &


# # Target domain shoaib4, source domain: all expect taregt + (5,6); 
# timestamp=$(date +%Y%m%d%H%M%S) 
# CroTyp="MCroDom"  
# GPUNum="2"  
# SrouceDomain="aet0" 
# TargetDomain="usc5"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# # Target domain shoaib4, source domain: all expect taregt + (5,6); 
# timestamp=$(date +%Y%m%d%H%M%S) 
# CroTyp="MCroDom"  
# GPUNum="3"  
# SrouceDomain="aet0" 
# TargetDomain="ku6"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &
