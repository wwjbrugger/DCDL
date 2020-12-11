#!/bin/bash
# bash run_set_of_experiments.sh -d numbers -n 1 -t 8
# first parameter is dataset [numbers, fashion, cifar]
# second parameter is number of repetitions
# third parameter is number threats

MAX_THREADS=8
REPETITIONS=1
OUTPUT_BASE_DIR=./terminal_output
ZIP_NAME=dev

while [ -n "$1" ]; do
  case "$1" in
  -d) DATASET="$2"; shift ;;
  -o) OUTPUT_BASE_DIR="$2"; shift ;;
  -n) REPETITIONS="$2"; shift ;;
  -t) MAX_THREADS="$2"; shift ;;
 # -l) LABEL="$2"; shift ;;
  -z) ZIP_NAME="$2"; shift ;;
  --) break ;;
  *) echo "Option $1 not recognized" ;;
  esac
  shift
done


if [ -z $DATASET ] ; then
  echo "You have to choose an dataset"
  exit
fi

echo "using output dir $OUTPUT_BASE_DIR"
echo "using number of threads $MAX_THREADS"
echo "using number of repetitions per thread $REPETITIONS"
echo "using zip name $ZIP_NAME"
echo "using Label $LABEL"

OUTPUT_SUBDIR=$(date +%F)
OUTPUT_DIR=$OUTPUT_BASE_DIR/$OUTPUT_SUBDIR
echo "outputting to $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"


for ((repetiton=0 ; repetiton<$REPETITIONS ; repetiton++)); do
	for ((label=0 ; label<9 ; label++)); do
		echo repetition $repetition label $label
		 PREFIX="$DATASET-l$label-r$repetiton-$(date +%Y-%m-%dT%H-%M)"
		 PYTHONPATH="$PWD/.." OMP_NUM_THREADS=$MAX_THREADS python start.py $DATASET $label >"$OUTPUT_DIR/$PREFIX-out.log" 2>"$OUTPUT_DIR/$PREFIX-err.log"
	done
done

ZIP_PATH="../terminal_out"
ZIP_NAME="$ZIP_NAME-$(date +%F).zip"
echo "zipping output to $ZIP_NAME"
rm -f "$ZIP_NAME"
pushd "$OUTPUT_BASE_DIR" || exit
mkdir -p $ZIP_PATH
zip -r "$ZIP_PATH/$ZIP_NAME" "$OUTPUT_SUBDIR"
popd || exit

# clean up
rm -rd "$OUTPUT_BASE_DIR"


