from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import itertools

from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import torch
import numpy as np
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
# import rocal_pybind.tensor
import sys
import cv2
import os
from math import sqrt
import ctypes

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

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False, num_anchors=8732):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        print("INIT")
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
        self.display = True

        #Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        # Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        print("INIT exit")

    def next(self):
        return self.__next__()

    def __next__(self):
        print("next")
        if(self.loader.isEmpty()):
            raise StopIteration
        print("here 1")
        if self.loader.rocalRun() != 0:
            print("here")
            raise StopIteration
        else:
            print("Comes to Next in COCO pipeline ")
            self.output_tensor_list = self.loader.rocalGetOutputTensors()

        #From init

        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels
        print(self.output_tensor_list)
        self.w = self.output_tensor_list[0].batch_width()
        self.h = self.output_tensor_list[0].batch_height()
        self.bs = self.output_tensor_list[0].batch_size()
        print("\n Batch Size",self.bs)
        self.color_format = self.output_tensor_list[0].color_format()

        #NHWC default for now
        self.out = torch.empty((self.bs, self.h, self.w, self.color_format,), dtype=torch.uint8)
        self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))


        # 1D labels & bboxes array
        torch_gpu_device = torch.device('cuda', self.device_id)
        labels_array, boxes_array = self.loader.getEncodedBoxesAndLables(self.bs, int(self.num_anchors))
        self.encoded_bboxes = torch.as_tensor(boxes_array, dtype=torch.float32, device=torch_gpu_device)
        self.encoded_bboxes = self.encoded_bboxes.view(self.bs, self.num_anchors, 4)
        self.encoded_labels = torch.as_tensor(labels_array, dtype=torch.int32, device=torch_gpu_device)
        encoded_bboxes_tensor = self.encoded_bboxes.cpu()
        encodded_labels_tensor = self.encoded_labels.cpu()

        # Image id of a batch of images
        self.loader.GetImageId(self.image_id)

        # Image sizes of a batch
        self.loader.GetImgSizes(self.img_size)

        # print(encoded_bboxes_tensor)
        # print(encodded_labels_tensor.shape)
        # exit(0)
        image_id_tensor = torch.tensor(self.image_id, device=torch_gpu_device)
        image_size_tensor = torch.tensor(self.img_size, device=torch_gpu_device).view(-1, self.bs, 2)
        print("Image ID :",image_id_tensor)
        print("Image SIZE :",image_size_tensor)
        # exit(0)
        for i in range(self.bs):
            index_list = []
            actual_bboxes = []
            actual_labels = []
            for idx, x in enumerate(encodded_labels_tensor[i]):
                if x != 0:
                    index_list.append(idx)
                    actual_bboxes.append(encoded_bboxes_tensor[i][idx].tolist())
                    actual_labels.append(encodded_labels_tensor[i][idx].tolist())
        print(actual_labels)
        print(actual_bboxes)
        # exit(0)

        if self.display:
            img = self.out
            draw_patches(img[i], self.image_id[i],
                            actual_bboxes, self.device)
        # exit(0)
        return (self.out), encoded_bboxes_tensor, encodded_labels_tensor, image_id_tensor, image_size_tensor
        return self.out

    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img, idx, bboxes, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    htot, wtot, _ = img.shape

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (xc, yc, w, h) in bboxes:
        l = xc - 0.5*(w)
        t = yc - 0.5*(h)
        r = xc + 0.5*(w)
        b = yc + 0.5*(h)
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]*wtot), int(loc_[1] * htot)), (int(
            (loc_[2] * wtot)), int((loc_[3] * htot))), color, thickness)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER/" + str(idx)+"_"+"train"+".png", image)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_path annotation_path cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    image_path = sys.argv[1]
    annotation_path = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[4])
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

    # Anchors
    def coco_anchors():  # Should be Tensor of floats in ltrb format - input - Mx4 where M="No of anchor boxes"
        fig_size = 300
        feat_size = [38, 19, 10, 5, 3, 1]
        steps = [8, 16, 32, 64, 100, 300]

        # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
        scales = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        default_boxes = []
        fk = fig_size/np.array(steps)
        # size of feature and number of feature
        for idx, sfeat in enumerate(feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    default_boxes.append((cx, cy, w, h))
        dboxes = torch.tensor(default_boxes, dtype=torch.float)
        dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        dboxes_ltrb = dboxes.clone()
        dboxes_ltrb[:, 0] = dboxes[:, 0] - 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 1] = dboxes[:, 1] - 0.5 * dboxes[:, 3]
        dboxes_ltrb[:, 2] = dboxes[:, 0] + 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 3] = dboxes[:, 1] + 0.5 * dboxes[:, 3]

        return dboxes_ltrb
    default_boxes = coco_anchors().numpy().flatten().tolist()
    print("*********************************************************************")

    coco_train_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)

    with coco_train_pipeline:
        jpegs, bboxes, labels = fn.readers.coco(annotations_file=annotation_path, random_shuffle=False, seed=random_seed, is_box_encoder=True)
        crop_begin, crop_size, bboxes, labels = fn.random_bbox_crop(bboxes, labels,
                                                                    device="cpu",
                                                                    aspect_ratio=[
                                                                        0.5, 2.0],
                                                                    thresholds=[
                                                                        0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                                                    scaling=[
                                                                        0.3, 1.0],
                                                                    bbox_layout="xyXY",
                                                                    allow_no_crop=True,
                                                                    num_attempts=50)
        images_decoded = fn.decoders.image_slice(jpegs, crop_begin, crop_size, device="mixed", output_type=types.RGB, file_root=image_path,
                                                 annotations_file=annotation_path, random_shuffle=False, seed=random_seed, num_shards=world_size, shard_id=local_rank)
        res = fn.resize(images_decoded, resize_width=224, resize_height=224, rocal_tensor_layout = types.NHWC, rocal_tensor_output_type = types.UINT8)
        saturation = fn.uniform(rng_range=[0.5, 1.5])
        contrast = fn.uniform(rng_range=[0.5, 1.5])
        brightness = fn.uniform(rng_range=[0.875, 1.125])
        hue = fn.uniform(rng_range=[-0.5, 0.5])
        ct_images = fn.color_twist(
            res, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(ct_images, device="gpu",
                                            rocal_tensor_layout = types.NHWC,
                                            rocal_tensor_output_type = types.UINT8,
                                            crop=(224, 224),
                                            mirror=0,
                                            image_type=types.RGB,
                                            mean=[0,0,0],
                                            std=[1,1,1])
        _, _ = fn.box_encoder(bboxes, labels,
                                  criteria=0.5,
                                  anchors=default_boxes)
        coco_train_pipeline.set_outputs(cmnp)

    coco_train_pipeline.build()
    imageIteratorPipeline = ROCALCOCOIterator(coco_train_pipeline)
    cnt = 0
    for i , it in enumerate(imageIteratorPipeline):
        print("************************************** i *************************************",i)
        # for img in it[0]:
        #     print(img.shape)
        #     cnt = cnt + 1
        #     draw_patches(img, cnt, "cpu")

    print("*********************************************************************")



    exit(0)


    import timeit
    start = timeit.default_timer() #Timer starts



    stop = timeit.default_timer() #Timer Stops
    print('\n Time: ', stop - start)

if __name__ == '__main__':
    main()
