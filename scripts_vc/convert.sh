#!/bin/bash
if [ "$#" -lt "2" ]
then
	echo "usage: bash $0 source_path target_path [--no_f0]"
	exit 1
fi
source="$1"
target="$2"
mkdir tmp
mkdir output
mkdir tmp/target
mkdir tmp/files
ln "$target" "tmp/target"

# extract f0
python3 f0_extract_avg/f0_extract.py --source_wav_dir tmp/target --output_dir tmp/files

# extract xvector
python3 xv_extract/inference.py --input "$target" --xv_dir tmp/files --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze

# generation
python3 generate_xv_f0/inference_single.py --input "$source" --output_dir output --xv_path `ls tmp/files/*xvector` --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze/ --f0_std --f0_log $3
rm -r tmp
name="`basename "$source"`"
echo "output: output/${name%.*}_gen.${name##*.}"
