#!/bin/bash

url="https://data.uni-hannover.de/vault/ikg/yuan/cosense3d/"

# Check if the input file containing URLs is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Dataset name: OPV2Vt | OPV2Vt> $1 <Out dir>"
    exit 1
fi

# Input file with the list of URLs
url_dir=$url$1
output_dir=$2

# Check if the output directory exists, if not, create it
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

pip install gdown

if [ "$1" == "OPV2Vt" ]; then
#  files=("opv2vt_meta.zip" "opv2vt_test.zip") # train files are broken
  files=("opv2vt_meta.zip" "opv2vt_test.zip" "opv2vt_train.zip" "opv2vt_train.z01" "opv2vt_train.z02" "opv2vt_train.z03")
  for f in "${files[@]}"; do
      url_file="$url_dir/$f"
      wget -P "$output_dir" "$url_file"
  done

  cd "$output_dir"
#  gdown --id 1oSPBZfgRHFCSEQjlVU3TLUwaKA5WYEbF
  cat opv2vt_train.z01 opv2vt_train.z02 opv2vt_train.z03 opv2vt_train.zip  > train.zip
  unzip train.zip
  rm train.zip
  rm opv2vt_train.z01 opv2vt_train.z02 opv2vt_train.z03 opv2vt_train.zip train.zip
  unzip opv2vt_test.zip
  rm opv2vt_test.zip
  unzip opv2vt_meta.zip
  mv opv2vt_meta meta
  rm opv2vt_meta.zip


elif [ "$1" == "OPV2Va" ]; then
 # wget -P "$output_dir" https://data.uni-hannover.de/dataset/678827e9-bb64-44b8-b8fd-e583c740b5f5/resource/eade1879-e67b-4112-a088-2a92ca76e004/download/opv2v_meta.zip
  files=("test.z01" "test.zip" "train.z01" "train.z02" "train.z03" "train.z04" "train.z05" "train.z06" "train.zip" )
  for f in "${files[@]}"; do
      url_file="$url_dir/$f"
      
      wget -P "$output_dir" "$url_file"
  done

  cd "$output_dir"
  cat train.z01 train.z02 train.z03 train.z04 train.z05 train.z06 train.zip   > combined.zip
  unzip combined.zip
  rm train.z01 train.z02 train.z03 train.z04 train.z05 train.z06 train.zip combined.zip
  cat test.z01 test.zip   > combined.zip
  unzip combined.zip
  rm test.z01 test.zip combined.zip
  unzip opv2v_meta.zip
  mv opv2v_full meta
  rm opv2v_meta.zip
fi

echo "Download completed."


