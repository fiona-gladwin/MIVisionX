#!/bin/bash

################ AVAILABLE READERS ######################
##   DEFAULT        : VideoReader                      ##
##   READER_CASE 2  : VideoReaderResize                ##
##   READER_CASE 3  : SequenceReader                   ##
#########################################################
INPUT_PATH=$1
READER_CASE=$2

if [ -z "$INPUT_PATH" ]
  then
    echo "No input argument supplied"
    exit
fi

# Handles relative input path
if [ ! -d "$INPUT_PATH" & [[ "$INPUT_PATH" != /* ]]
then
  CWD=$(pwd)
  INPUT_PATH="$CWD/$INPUT_PATH"
fi

if [ -z "$READER_CASE" ]
  then
    READER_CASE=1
fi

# Building video unit test
sudo rm -rvf build*
mkdir build
cd build || exit
cmake ..
make

# Arguments used in video unit test
SAVE_FRAMES=1   # (save_frames:on/off)
RGB=1           # (rgb:1/gray:0)
DEVICE=1        # (cpu:0/gpu:1)
HARDWARE_DECODE_MODE=0 # (hardware_decode_mode:on/off)
SHUFFLE=0       # (shuffle:on/off)

BATCH_SIZE=1         # Number of sequences per batch
SEQUENCE_LENGTH=10    # Number of frames per sequence
STEP=9               # Frame interval from one sequence to another sequence
STRIDE=1             # Frame interval within frames in a sequences
RESIZE_WIDTH=960    # width with which frames should be resized (applicable only for READER_CASE 2)
RESIZE_HEIGHT=540    # height with which frames should be resized (applicable only for READER_CASE 2)

FILELIST_FRAMENUM=1          # enables file number or timestamps parsing for text file input

echo ./rocAL_optical_flow "$INPUT_PATH" $READER_CASE $DEVICE $HARDWARE_DECODE_MODE $BATCH_SIZE $SEQUENCE_LENGTH $STEP $STRIDE \
$RGB $SAVE_FRAMES $SHUFFLE $RESIZE_WIDTH $RESIZE_HEIGHT $FILELIST_FRAMENUM

./rocAL_optical_flow "$INPUT_PATH" $READER_CASE $DEVICE $HARDWARE_DECODE_MODE $BATCH_SIZE $SEQUENCE_LENGTH $STEP $STRIDE \
$RGB $SAVE_FRAMES $SHUFFLE $RESIZE_WIDTH $RESIZE_HEIGHT $FILELIST_FRAMENUM
