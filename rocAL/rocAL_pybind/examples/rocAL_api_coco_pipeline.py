# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os
import math
import ctypes
import numpy as np
import random
import struct

class ROCALCOCOIterator(object):
    """
    COCO ROCAL iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False, num_anchors=120087):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.num_anchors = num_anchors
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")

    def next(self):
        return self.__next__()

    def __next__(self):
        if(self.loader.isEmpty()):
            raise StopIteration
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.getOutputTensors()
        if self.out is None:
            self.w, self.h = self.output_tensor_list[0].max_shape()
            self.bs = self.output_tensor_list[0].batch_size()
            self.color_format = self.output_tensor_list[0].color_format() if self.color_format else self.color_format
            torch_gpu_device = torch.device('cuda', self.device_id)
            if self.tensor_format == types.NCHW:
                torch_gpu_device = torch.device('cuda', self.device_id)
                if self.tensor_dtype == types.FLOAT:
                    self.out = torch.empty((self.batch_size, self.color_format, self.h, self.w,), dtype = torch.float32, device = torch_gpu_device)
                elif self.tensor_dtype == types.FLOAT16:
                    self.out = torch.empty((self.batch_size, self.color_format, self.h, self.w,), dtype = torch.float16, device = torch_gpu_device)
                elif self.tensor_dtype == types.UINT8:
                    self.out = torch.empty((self.batch_size, self.color_format, self.h, self.w,), dtype = torch.uint8, device = torch_gpu_device)
            else:  # NHWC
                torch_gpu_device = torch.device('cuda', self.device_id)
                if self.tensor_dtype == types.FLOAT:
                    self.out = torch.empty((self.batch_size, self.h, self.w, self.color_format), dtype = torch.float32, device = torch_gpu_device)
                elif self.tensor_dtype == types.FLOAT16:
                    self.out = torch.empty((self.batch_size, self.h, self.w, self.color_format), dtype = torch.float16, device = torch_gpu_device)
                elif self.tensor_dtype == types.UINT8:
                    self.out = torch.empty((self.batch_size, self.h, self.w, self.color_format), dtype = torch.uint8, device = torch_gpu_device)
            self.labels_tensor = torch.empty(self.batch_size, dtype = torch.int32, device = torch_gpu_device)

        self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))

        labels_array = self.loader.GetBoundingBoxLabels()
        encodded_labels_tensor = []
        encoded_bboxes_tensor = []
        for label in labels_array:
            self.encoded_labels = torch.as_tensor(label, dtype=torch.int64)
            encodded_labels_tensor.append(self.encoded_labels)

        boxes_array = self.loader.GetBoundingBoxCords()
        for box in boxes_array:
            self.encoded_bboxes = torch.as_tensor(box, dtype=torch.float16)
            self.encoded_bboxes = self.encoded_bboxes * 800
            self.encoded_bboxes = self.encoded_bboxes.view(-1, 4)
            encoded_bboxes_tensor.append(self.encoded_bboxes)

        matched_idxs = self.loader.GetMatchedIndices()
        self.matched_idxs = torch.as_tensor(matched_idxs, dtype=torch.int64)
        matched_idxs_tensor = self.matched_idxs.view(-1, 120087)

        # Image sizes and Image id of a batch
        self.loader.GetImgSizes(self.img_size)
        image_size_tensor = torch.tensor(self.img_size).view(self.bs, 3)

        targets = { 'boxes' : encoded_bboxes_tensor,
                    'labels' : encodded_labels_tensor,
                    'image_id' : image_size_tensor[:, 2:3].cuda(),
                    'original_image_size' : image_size_tensor[:, 0:2].cuda(),
                    'matched_idxs' : matched_idxs_tensor
        }
        return self.out, targets

    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img, idx, bboxes, device="cpu"):
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    image = image.transpose([1, 2, 0])

    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
            (loc_[2])), int((loc_[3]))), color, thickness)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/COCO_READER/" + str(idx)+"_"+"train"+".png", image * 255)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_path annotation_path cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/COCO_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    image_path = sys.argv[1]
    annotation_path = sys.argv[2]
    device = sys.argv[3]
    _rali_cpu = True if device == "cpu" else False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=800
    local_rank = 0
    world_size = 1
    # Anchors - load default anchors from a binary file  
    with open('Default_anchors_retinanet.bin', 'rb') as f_read:
        anchor_bytes = f_read.read()
    num_anchors = len(anchor_bytes) // 4
    anchor_list = []
    for i in range(num_anchors):
        anchor_bytes_single = anchor_bytes[i * 4: (i + 1) * 4]
        anchor_float = struct.unpack('!f', anchor_bytes_single)[0]
        anchor_list.append(anchor_float)
    print("*********************************************************************")
    coco_train_pipeline = Pipeline(batch_size = batch_size, num_threads = num_threads, device_id = device_id,seed = random_seed, rocal_cpu = _rali_cpu, tensor_layout = types.NCHW)
    with coco_train_pipeline:
        jpegs, bboxes, labels = fn.readers.coco(file_root=image_path,
                                                 annotations_file=annotation_path, 
                                                 random_shuffle=True,
                                                 shard_id=local_rank, 
                                                 num_shards=world_size,
                                                 seed=random_seed, 
                                                 is_box_encoder=False,
                                                 is_box_iou_matcher=True)

        print("*********************** SHARD ID ************************",local_rank)
        print("*********************** NUM SHARDS **********************",world_size)
        images_decoded = fn.decoders.image(jpegs, output_type = types.RGB, file_root=image_path, annotations_file=annotation_path, random_shuffle=False,shard_id=local_rank, num_shards=world_size)
        flip_coin = fn.random.coin_flip(probability=0.5)
        images = fn.resize_mirror_normalize(images_decoded, device="gpu",
                                            image_type=types.RGB,
                                            resize_width=crop, resize_height=crop,
                                            mirror=flip_coin,
                                            output_layout = types.NCHW,
                                            output_dtype = types.FLOAT,
                                            mean=[0,0,0],
                                            std=[1,1,1])
        matched_idxs  = fn.box_iou_matcher(anchors=anchor_list, criteria=0.5,
                                     high_threshold=0.5, low_threshold=0.4,
                                     allow_low_quality_matches=True)
        coco_train_pipeline.set_outputs(images)
    coco_train_pipeline.build()
    COCOIteratorPipeline = ROCALCOCOIterator(coco_train_pipeline)
    cnt = 0
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i, it in enumerate(COCOIteratorPipeline):
            print("************************************** i *************************************",i)
            print(it)
            for i, img in enumerate(it[0]):
                draw_patches(img, cnt, it[1]['boxes'][i], "cpu")
                cnt =  cnt + 1
        COCOIteratorPipeline.reset()
    print("*********************************************************************")

if __name__ == '__main__':
    main()
