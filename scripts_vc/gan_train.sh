#!/bin/bash
if [ "$#" -lt "2" ]
then
	echo "usage: bash $0 source_path target_path"
	exit 1
fi
source="$1"
target="$2"
mkdir tmp
mkdir output
mkdir tmp/target
mkdir tmp/files
ln "$target" "tmp/target"

echo "extract target f0"
python3 f0_extract_avg/f0_extract.py --source_wav_dir tmp/target --output_dir tmp/files

echo "extract target speaker vector"
python3 xv_extract/inference.py --input "$target" --xv_dir tmp/files --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze

echo "copy original model"
cp -r pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze/ pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze-gan-training-xv

# echo "create source list file"
echo "$source" > gan-training-xv/source.lst
echo "$source" >> gan-training-xv/source.lst

# echo "copy config file"
cp gan-training-xv/libri_tts_clean_100_fbank_xv_ssl_freeze.json pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze-gan-training-xv/config.json

echo "train GAN"
python3 gan-training-xv/train.py --checkpoint_path pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze-gan-training-xv/ --config gan-training-xv/libri_tts_clean_100_fbank_xv_ssl_freeze.json --training_epochs 100 --xv_path tmp/files/*.xvector --f0_path tmp/files/*.npy --validation_interval 1 --summary_interval 1 --stdout_interval 1

# echo "remove temporary files"
rm -r tmp
