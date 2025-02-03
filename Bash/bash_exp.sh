#!/bin/bash


# nohup python -u phase_ar_decen.py unihar merge 20_120 -g 0 -sd 1 -e unihar_bert_fed_d1 -s unihar_decen_d1 > log/phase_ar_decen_unihar_d1.log &

# Multiple source, single target domain. 
# Target domain 1, source domain: all expect taregt ; 
timestamp=$(date +%Y%m%d%H%M%S) 
GPUNum="5" 
CroTyp="MCroDom" # The cross dataset manner. 
modelP="dcnn_v1" # v3_limu; v1_unihar 
Feature_extractor_path="extractor/limu_bert_motion" # Only for source. 
SrouceDomain="aet0"  
TargetDomain="motion3" #'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
PretrainLogFileName="saved/log_0809/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
AdaptLogFileName="saved/log_0809/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &

# nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &


# timestamp=$(date +%Y%m%d%H%M%S) 
# GPUNum="0" 
# CroTyp="SCroDom" # The cross dataset manner. 
# modelP="gru_v3" # v3_limu; v1_unihar 
# Feature_extractor_path="extractor/limu_bert_motion" # Only for source. 
# SrouceDomain="uci2"  
# TargetDomain="usc5"  #'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
# nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# timestamp=$(date +%Y%m%d%H%M%S) 
# GPUNum="0" 
# CroTyp="SCroDom" # The cross dataset manner. 
# modelP="gru_v3" # v3_limu; v1_unihar 
# Feature_extractor_path="extractor/limu_bert_motion" # Only for source. 
# SrouceDomain="uci2"  
# TargetDomain="motion3"  #'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
# nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# timestamp=$(date +%Y%m%d%H%M%S) 
# GPUNum="5" 
# CroTyp="MCroDom" # The cross dataset manner. 
# modelP="gru_v3" # v3_limu; v1_unihar 
# Feature_extractor_path="extractor/limu_bert_motion" # Only for source. 
# SrouceDomain="uci2" 
# TargetDomain="shoaib4"  #'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6' 
# PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model $modelP --g $GPUNum --e $Feature_extractor_path --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName &
# nohup python -u cotta.py --dataset new_20_120 --model $modelP --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &
# # # Target domain 1, source domain: all expect taregt ; 
# # timestamp=$(date +%Y%m%d%H%M%S) 
# # GPUNum="0"
# # CroTyp="SCroDom" # The cross dataset manner. 
# # SrouceDomain="uci2" 
# # TargetDomain="motion3"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
# # PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# # # Target domain 1, source domain: all expect taregt ; 
# # timestamp=$(date +%Y%m%d%H%M%S) 
# # GPUNum="0"
# # CroTyp="SCroDom" # The cross dataset manner. 
# # SrouceDomain="uci2" 
# # TargetDomain="shoaib4"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
# # PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# # #########################
# # # Target domain 1, source domain: all expect taregt ; 
# # timestamp=$(date +%Y%m%d%H%M%S) 
# # GPUNum="0"
# # CroTyp="SCroDom" # The cross dataset manner. 
# # SrouceDomain="hhar1" 
# # TargetDomain="uci2"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
# # PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# # # Target domain 1, source domain: all expect taregt ; 
# # timestamp=$(date +%Y%m%d%H%M%S) 
# # GPUNum="0"
# # CroTyp="SCroDom" # The cross dataset manner. 
# # SrouceDomain="hhar1" 
# # TargetDomain="motion3"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
# # PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# # # Target domain 1, source domain: all expect taregt ; 
# # timestamp=$(date +%Y%m%d%H%M%S) 
# # GPUNum="0"
# # CroTyp="SCroDom" # The cross dataset manner. 
# # SrouceDomain="hhar1"  
# # TargetDomain="shoaib4"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
# # PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# # #########################
# # # Target domain 1, source domain: all expect taregt ; 
# # timestamp=$(date +%Y%m%d%H%M%S) 
# # GPUNum="0"
# # CroTyp="SCroDom" # The cross dataset manner. 
# # SrouceDomain="motion3" 
# # TargetDomain="hhar1"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
# # PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# # # Target domain 1, source domain: all expect taregt ; 
# # timestamp=$(date +%Y%m%d%H%M%S) 
# # GPUNum="0"
# # CroTyp="SCroDom" # The cross dataset manner. 
# # SrouceDomain="motion3" 
# # TargetDomain="uci2"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
# # PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &

# # # Target domain 1, source domain: all expect taregt ; 
# # timestamp=$(date +%Y%m%d%H%M%S) 
# # GPUNum="0"
# # CroTyp="SCroDom" # The cross dataset manner. 
# # SrouceDomain="motion3"  
# # TargetDomain="shoaib4"  # 'aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'
# # PretrainLogFileName="saved/log/Pretrain_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # AdaptLogFileName="saved/log/Adapt_"$SrouceDomain$TargetDomain"_"$timestamp".log" 
# # nohup python -u pretrain.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $PretrainLogFileName && python -u cotta.py --dataset new_20_120 --model dcnn_v1 --g $GPUNum --cd $CroTyp --SDom $SrouceDomain --TDom $TargetDomain > $AdaptLogFileName &
