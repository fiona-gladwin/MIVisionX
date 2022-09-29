echo This is my first shell script
folder=./
if test -d "$folder"; then 
    rm -rf ./output_folder
fi
mkdir output_folder
#########################

# sudo rm -rvf build*
# mkdir build
# cd build || exit
# cmake ..
# make

###########################
cd build
INPUTPATH=$1
width=$2
height=$3
testcase=0
batch_size=$4
device=$5
rgb=1
shard_count=1
shuffle=0


#############################################  PKD   #########################################
echo arguments $INPUTPATH $width $height $testcase $batch_size $device $shard_count $shuffle
./rocAL_performance_tests /media/MIVisionX-data/rocal_data/images_jpg/labels_folder/0/ 300 300 0 1 0 1 1 0 &> test1.txt
i=1

while [ $testcase -le 28 ]
do
    # echo **************************  $testcase ********************************
    file_name=$testcase.txt
	# aug_list = ["rocalResize", "rocalCropResize", "rocalRotate", "rocalBrightness", "rocalGamma", "rocalContrast", "rocalFlip", "rocalBlur", "rocalBlend", "rocalWarpAffine", "rocalFishEye", "rocalVignette", "rocalVignette", "rocalSnPNoise", "rocalSnow", "rocalRain", "rocalColorTemp", "rocalFog", "rocalLensCorrection", "rocalPixelate", "rocalExposure", "rocalHue", "rocalSaturation", "rocalCopy", "rocalColorTwist", "rocalCropMirrorNormalize", "rocalCrop", "rocalResizeCropMirror", "No-Op"]

    case "$testcase" in
   "0") file_name=rocalResize.txt
   ;;
   "1") file_name=rocalCropResize.txt
   ;;
   "2") file_name=rocalRotate.txt 
   ;;
   "3") file_name=rocalBrightness.txt
   ;;
   "4") file_name=rocalGamma.txt
   ;;
   "5") file_name=rocalContrast.txt 
   ;;
   "6") file_name=rocalFlip.txt
   ;;
   "7") file_name=rocalBlur.txt
   ;;
   "8") file_name=rocalBlend.txt 
   ;;
   "9") file_name=rocalWarpAffine.txt
   ;;
   "10") file_name=rocalFishEye.txt
   ;;
   "11") file_name=rocalVignette.txt 
   ;;
   "12") file_name=rocalJitter.txt
   ;;
   "13") file_name=rocalSnPNoise.txt
   ;;
   "14") file_name=rocalSnow.txt 
   ;;
   "15") file_name=rocalRain.txt
   ;;
   "16") file_name=rocalColorTemp.txt
   ;;
   "17") file_name=rocalFog.txt 
   ;;
   "18") file_name=rocalLensCorrection.txt
   ;;
   "19") file_name=rocalPixelate.txt
   ;;
   "20") file_name=rocalExposure.txt 
   ;;
   "21") file_name=rocalHue.txt
   ;;
   "22") file_name=rocalSaturation.txt
   ;;
   "23") file_name=rocalCopy.txt 
   ;;
   "24") file_name=rocalColorTwist.txt
   ;;
   "25") file_name=rocalCropMirrorNormalize.txt
   ;;
   "26") file_name=rocalCrop.txt 
   ;;
   "27") file_name=rocalResizeCropMirror.txt
   ;;
   "28") file_name=No-Op.txt
   ;;
esac
    echo arguments $INPUTPATH $width $height $testcase $batch_size $device $shard_count $shuffle >> ./../output_folder/$file_name
    ./rocAL_performance_tests $INPUTPATH $width $height $testcase $batch_size $device $rgb $shard_count $shuffle >> ./../output_folder/$file_name

    testcase=$(($testcase+1))
done
# ./rocAL_performance_tests $INPUTPATH $width $height $testcase $batch_size $device $rgb $shard_count $shuffle
echo End of my shell script