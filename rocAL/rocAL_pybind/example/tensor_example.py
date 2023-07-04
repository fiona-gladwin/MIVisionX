from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rocal.plugin.pytorch import ROCALClassificationIterator

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import cv2
import os

def draw_patches(img, idx, device, layout="NCHW", dtype="fp32"):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.cpu().numpy()
    if layout!= "NHWC":
        image = image.transpose([1, 2, 0])
    if dtype == "fp16":
        image = (image).astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness/" + str(idx)+"_"+"train"+".png", image*255)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

    local_rank = 0
    world_size = 3
    rali_cpu= True
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'mixed'

    image_classification_train_pipeline = Pipeline(batch_size = batch_size, num_threads = num_threads, device_id = device_id, seed = random_seed, rocal_cpu = _rali_cpu, tensor_layout = types.NHWC)
    with image_classification_train_pipeline:
        jpegs, labels = fn.readers.file(file_root = data_path)
        # decode = fn.decoders.image(jpegs, output_type = types.RGB,
        #                                 file_root = data_path, shard_id = local_rank, num_shards = world_size, random_shuffle = True)
        decode = fn.decoders.image_slice(jpegs, output_type=types.RGB,max_decoded_width=1000, max_decoded_height=1000,
                                                    file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        # res = fn.resize(decode, resize_width = 224, resize_height = 224, rocal_tensor_layout = types.NHWC, rocal_tensor_output_type = types.UINT8)
        res = fn.resize(decode, scaling_mode=types.SCALING_MODE_NOT_SMALLER, interpolation_type=types.TRIANGULAR_INTERPOLATION, resize_shorter=256, rocal_tensor_output_type = types.UINT8)

        # flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(res, device="gpu",
                                            rocal_tensor_layout = types.NCHW,
                                            rocal_tensor_output_type = types.FLOAT,
                                            crop=(224, 224),
                                            mirror=0,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        image_classification_train_pipeline.set_outputs(cmnp)

    image_classification_train_pipeline.build()
    imageIteratorPipeline = ROCALClassificationIterator(image_classification_train_pipeline, device=sys.argv[2], device_id=local_rank,  )
    cnt = 0
    for epoch in range(3):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(imageIteratorPipeline):
            print(it)
            print("************************************** i *************************************",i)
            for img in it[0]:
                print("Image shape :: ", img.shape)
                # print("Image :: ", img)
                cnt = cnt + 1
                draw_patches(img, cnt, "cpu",  layout="NCHW")
        imageIteratorPipeline.reset()
    print("*********************************************************************")

if __name__ == '__main__':
    main()

