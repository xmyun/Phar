#!/bin/bash


# nohup python -u phase_ar_decen.py unihar merge 20_120 -g 0 -sd 1 -e unihar_bert_fed_d1 -s unihar_decen_d1 > log/phase_ar_decen_unihar_d1.log &

# Multiple source, single target domain. 
# Target domain 1, source domain: all expect taregt ; 
timestamp=$(date +%Y%m%d%H%M%S) 
SrouceDomain="uci2" 
TargetDomain="motion3"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 -g 0 --cd SCroDom --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName 
# && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --cd SCroDom --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &


# # Target domain 5, source domain: all expect taregt; 
# timestamp=$(date +%Y%m%d%H%M%S)
# SrouceDomain="aet0"
# TargetDomain="usc5"
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log"
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log"
# nohup python pretrain.py --model dcnn_v1 -g 0 --dataset shoaib_20_120 --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &&\
# python cotta.py --model dcnn_v1 -g 0 --dataset shoaib_20_120 --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# ###############################################
# ###############################################

# # Target domain 1, source domain: 5 ; 
# timestamp=$(date +%Y%m%d%H%M%S)
# SrouceDomain="hhar1"
# TargetDomain="usc5"
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log"
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log"
# nohup python pretrain.py --model dcnn_v1 --dataset shoaib_20_120 --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &&\
# python cotta.py --model dcnn_v1 --dataset shoaib_20_120 --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &
