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

def draw_patches(img, idx, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
            image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    image = image.transpose([1, 2, 0])
    # print(img.shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness" + "/" + str(idx)+"_"+"train"+".png", image * 255)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness"
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

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)
    local_rank = 0
    world_size = 1
    rali_cpu= True
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    # with pipe:
    #     jpegs, labels = fn.readers.file(file_root=data_path)
    #     images = fn.decoders.image(jpegs,file_root=data_path, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    #     brightend_images = fn.brightness(images)
    #     # brightend_images2 = fn.brightness(brightend_images)

    #     pipe.set_outputs(brightend_images)

    # pipe.build()
    # imageIterator = ROCALClassificationIterator(pipe)
    # cnt = 0
    # for i , it in enumerate(imageIterator):
    #     print("************************************** i *************************************",i)
    #     for img in it[0]:
    #         print(img.shape)
    #         cnt = cnt + 1
    #         draw_patches(img, cnt, "cpu")

    print("*********************************************************************")

    image_classification_train_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)

    with image_classification_train_pipeline:
        jpegs, labels = fn.readers.file(file_root=data_path)
        decode = fn.decoders.image_slice(jpegs, output_type=types.RGB,
                                        file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        res = fn.resize(decode, resize_width=224, resize_height=224, rocal_tensor_layout = types.NHWC, rocal_tensor_output_type = types.UINT8)
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(res, device="gpu",
                                            rocal_tensor_layout = types.NHWC,
                                            rocal_tensor_output_type = types.FLOAT,
                                            crop=(224, 224),
                                            mirror=flip_coin,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        image_classification_train_pipeline.set_outputs(cmnp)

    image_classification_train_pipeline.build()
    imageIteratorPipeline = ROCALClassificationIterator(image_classification_train_pipeline)
    cnt = 0
    for epoch in range(3):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(imageIteratorPipeline):
            print(it)
            print("************************************** i *************************************",i)
            # for img in it[0]:
                # print(img.shape)
                # cnt = cnt + 1
                # draw_patches(img, cnt, "cpu")
        imageIteratorPipeline.reset()
    print("*********************************************************************")
    exit(0)
    # image_classification_val_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)

    # with image_classification_val_pipeline:
    #     jpegs, labels = fn.readers.file(file_root=data_path)
    #     decode = fn.decoders.image(jpegs,file_root=data_path, output_type=types.RGB, shard_id=local_rank, num_shards=world_size, random_shuffle=False)
    #     res = fn.resize_shorter(decode, resize_size = 256)
    #     centrecrop = fn.centre_crop(res, crop=(224, 224))
    #     cmnp = fn.crop_mirror_normalize(centrecrop, device="gpu",
    #                                         rocal_tensor_layout = types.NHWC,
    #                                         rocal_tensor_output_type = types.UINT8,
    #                                         crop=(224, 224),
    #                                         mirror=0,
    #                                         image_type=types.RGB,
    #                                         mean=[0.485 * 255,0.456 * 255,0.406 * 255],
    #                                         std=[0.229 * 255,0.224 * 255,0.225 * 255])
    #     image_classification_val_pipeline.set_outputs(cmnp)

    # image_classification_val_pipeline.build()
    # imageIteratorPipeline = ROCALClassificationIterator(image_classification_train_pipeline)
    # cnt = 0
    # for e in range(3):
    #     for i , it in enumerate(imageIteratorPipeline):
    #         print("************************************** i *************************************",i)
    #         for img in it[0]:
    #             # print(img.shape)
    #             cnt = cnt + 1
    #             draw_patches(img, cnt, "cpu")
    #     imageIteratorPipeline.reset()




    exit(0)

    # Need to call pipe.run() instead of iterator now (pipe.run() name is changed to pipe.run_tensor())
    output_data_batch_1 = pipe.run()
    print("----------------------")
    print(len(output_data_batch_1))
    # labels = pipe.rocalGetImageLabels()
    # print(labels)
    # output_data_batch_1 = pipe.run()
    # print("----------------------")
    # print(len(output_data_batch_1))

    # print("\n OUTPUT DATA!!!!: ", output_data_batch_1) # rocalTensorList 1
    exit(0)
    print("\n rocalTensor:: ",output_data_batch_1[0])
    # print("\n rocalTensor:: ",output_data_batch_1[0].at(0))
    print("\n size of rocalTensor ",output_data_batch_1[0].at(0).shape)

    print(output_data_batch_1[0].at(0).transpose(2,1,0).shape)
    print(output_data_batch_1[0].at(0))

    cv2.imwrite("output_images0_0.jpg", cv2.cvtColor(output_data_batch_1[0].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images0_1.jpg", cv2.cvtColor(output_data_batch_1[0].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images0_2.jpg", cv2.cvtColor(output_data_batch_1[0].at(2), cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images1_0.jpg", cv2.cvtColor(output_data_batch_1[1].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1_1.jpg", cv2.cvtColor(output_data_batch_1[1].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1_2.jpg", cv2.cvtColor(output_data_batch_1[1].at(2), cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images2_0.jpg", cv2.cvtColor(output_data_batch_1[2].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_1.jpg", cv2.cvtColor(output_data_batch_1[2].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_2.jpg", cv2.cvtColor(output_data_batch_1[2].at(2), cv2.COLOR_RGB2BGR))

    output_data_batch_2 = pipe.run()
    print("\n OUTPUT DATA BATCH 2!!!!: ", output_data_batch_2) # rocalTensorList 2

    cv2.imwrite("output_images2_0_0.jpg", cv2.cvtColor(output_data_batch_2[0].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_0_1.jpg", cv2.cvtColor(output_data_batch_2[0].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_0_2.jpg", cv2.cvtColor(output_data_batch_2[0].at(2), cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images2_1_0.jpg", cv2.cvtColor(output_data_batch_2[1].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_1_1.jpg", cv2.cvtColor(output_data_batch_2[1].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_1_2.jpg", cv2.cvtColor(output_data_batch_2[1].at(2), cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images2_2_0.jpg", cv2.cvtColor(output_data_batch_2[2].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_2_1.jpg", cv2.cvtColor(output_data_batch_2[2].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_2_2.jpg", cv2.cvtColor(output_data_batch_2[2].at(2), cv2.COLOR_RGB2BGR))

    output_data_batch_3 = pipe.run()
    print("3 BATCH")
    output_data_batch_4 = pipe.run()
    print("4 BATCH")
    output_data_batch_5 = pipe.run()
    print(output_data_batch_5)

    import timeit
    start = timeit.default_timer() #Timer starts



    stop = timeit.default_timer() #Timer Stops
    print('\n Time: ', stop - start)

if __name__ == '__main__':
    main()
