#!/bin/bash
if [ "$#" != "2" ]
then
    echo "usage: bash $0 (libri_dev|libri_test|vctk_dev|vctk_test) (trials_m[_all]|trials_f[_all])"
    exit 1
fi

subdataset="$1"
category="$2"

data_path="data/$subdataset/wav"
list_file="scp/vpc/${subdataset}_$category.lst"

home="conversion/$subdataset/$category"
f0_dir="$home/f0"

rm -r "$f0_dir"
mkdir -p "$f0_dir"

# echo "extracting f0s"
python f0_extract_avg/f0_extract.py --input_test_file "$list_file" --output_dir "$f0_dir"
ls "$f0_dir" | wc -l

# echo "creating speaker directories"
for x in `ls "$f0_dir"`
do
    spk=`echo "$x" | cut -d '-' -f 1 | cut -d '_' -f 1`
    mkdir -p "$f0_dir/$spk"
    mv "$f0_dir/$x" "$f0_dir/$spk/"
done
# echo "averaging speaker f0s"
for spk in `ls "$f0_dir"`
do
    python f0_extract_avg/f0_avg.py --input "$f0_dir/$spk" --output "$f0_dir/$spk.npy"
done
# echo "moving avgs"
mkdir "$f0_dir/avgs"
find "$f0_dir" -maxdepth 1 -name '*npy' -exec mv '{}' "$f0_dir/avgs" ';'
# echo "temporary subfolders (except avgs) of $f0_dir can be manually removed"
