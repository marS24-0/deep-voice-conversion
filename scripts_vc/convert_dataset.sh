#!/bin/bash
if [ "$#" -lt "2" ]
then
	echo "usage: bash $0 (libri_dev|libri_test|vctk_dev|vctk_test) (trials_m|trials_f) [--no_f0]"
	exit 1
fi

subdataset="$1"
category="$2"

dataset="${subdataset%%_*}" # extracts string before _

if [ "$dataset" = "vctk" ] # if vctk_dev or vctk_test, path suffixes are needed
then
    suffix1='_all'
    suffix2='_mic2'
else
    suffix1=''
    suffix2=''
fi

home="conversion/$subdataset/$category"
xv_dir="$home/xvectors"
xv_avg="$home/xv_avgs"

lst="scp/vpc/${subdataset}_${category}${suffix1}.lst"
trials_file="data/$subdataset/${category}${suffix2}"

if [ "$3" = '--no_f0' ]
then
    output="$home/gen/no_f0/output"
    rm -r "$home/gen/no_f0"
else
    output="$home/gen/with_f0/output"
    rm -r "$home/gen/with_f0"
fi

mkdir -p "$output"

# speaker vector
if [ ! -d "$xv_dir" ] || [ ! -d "$xv_avg" ] # if speaker vectors not already extracted
then
    mkdir -p "$xv_dir"
    mkdir -p "$xv_avg"

    echo "extract speaker vectors"
    python3 xv_extract/inference.py --input "$lst" --xv_dir "$xv_dir" --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze

    echo "average speaker vectors"
    python3 xv_extract/xvector_mean.py --input "$xv_dir" --output "$xv_avg" --dataset "$dataset"
fi

# F0
if [ ! -d "$home/f0" ] && [ "$3" != "--no_f0" ] # if f0 to be adapted and not already extracted
then
    echo "extract and average f0"
    bash f0_extract_avg/f0.sh "$subdataset" "${category}${suffix1}"
fi

# generation
echo "generate audios"
python3 generate_xv_f0/inference.py --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze/ --f0_std --f0_log $3 --input "$lst" --input_trials_file "$trials_file" --output_dir "$output" --xv_path "$xv_avg" --f0_path "$home/f0/avgs/"

echo "outputs generated in $output"
