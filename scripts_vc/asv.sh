#!/bin/bash
if [ "$#" -lt "2" ]
then
	echo "usage: bash $0 (libri_dev|libri_test|vctk_dev|vctk_test) (trials_m|trials_f]) [--original] [--no_f0]"
	exit 1
fi

subdataset="$1"
category="$2"

dataset="${subdataset%%_*}"

if [ "$dataset" = "vctk" ] # if vctk_dev or vctk_test
then
    suffix2='_mic2'
else
    suffix2=''
fi

home="conversion/$subdataset/$category"
xv_enrolls="$home/enrolls"

# parse arguments
original="0"
no_f0="0"
for arg in "$@"; do
  shift
  case "$arg" in
    '--original') original="1"   ;;
    '--no_f0') no_f0="1"   ;;
  esac
done

if [ "$no_f0" = "1" ]
then
	f0='no_f0'
else
	f0='with_f0'
fi

home_gen="$home/gen/$f0"
xv_gen="$home_gen/xv_gen"
mkdir -p "$home/det_rates"

# enrolls speaker vectors
if [ ! -d "$xv_enrolls" ] # if enrolls speaker vectors not already extracted
then
	mkdir -p "$xv_enrolls"
	echo "extract speaker vectors from enrolls"
	python3 xv_extract/inference.py --input "scp/vpc/${subdataset}_enrolls.lst" --xv_dir "$xv_enrolls" --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze
fi

if [ "$original" = 1 ]
then
	# ASV
	echo "run ASV"
	python3 speaker_verification.py --enrolls "data/$subdataset/enrolls$suffix2" --enrolls_root "$xv_enrolls" --trials "data/$subdataset/${category}${suffix2}" --trials_root "$home/xvectors" --dataset "$dataset" --rates_file "$home/det_rates/original.npz"
else
	rm -r "$xv_gen"
	mkdir -p "$xv_gen"

	# edit trials file, updating filenames to include source filename and target id in output filename
	echo "edit trials"
	python scripts_vc/edit_trials_file.py "data/$subdataset/${category}${suffix2}" "$home_gen/$category"

	# output speaker vectors
	echo "extract speaker vectors from generated audios"
	python3 xv_extract/inference.py --input "$home_gen/output" --xv_dir "$xv_gen" --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze

	# ASV
	echo "run ASV"
	python3 speaker_verification.py --enrolls "data/$subdataset/enrolls$suffix2" --enrolls_root "$xv_enrolls" --trials "$home_gen/$category" --trials_root "$xv_gen" --dataset "$dataset" --rates_file "$home/det_rates/converted_$f0.npz"
fi

# echo "the rates file is in $home/det_rates"
