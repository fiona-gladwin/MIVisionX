#Just a random comment

import types
import collections
import numpy as np
from random import shuffle
from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import cv2

def main():

    batch_size = 5
    data_dir = "/media/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/" # Pass a directory
    device = "cpu"
    def draw_patches(image, idx, device):
    #image is expected as a tensor, bboxes as numpy
        import cv2
        image = image.detach().numpy()
        image = image.transpose([1, 2, 0]) # NCHW
        image = (image).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(idx)+"_"+"train"+".png", image)

    #Define the Data Source for all image samples
    class ExternalInputIteratorMode2(object):
        def __init__(self, batch_size):
            self.images_dir = data_dir
            self.batch_size = batch_size
            self.files = []
            self.maxHeight = self.maxWidth = 0
            import os, glob
            for filename in glob.glob(os.path.join(self.images_dir, '*.jpg')):
                self.files.append(filename)
            shuffle(self.files)
            self.i = 0
            self.n = len(self.files)
            for x in range(self.n):
                jpeg_filename = self.files[x]
                label = 1
                image = cv2.imread(jpeg_filename, cv2.IMREAD_COLOR)
                # Check if the image was loaded successfully
                if image is None:
                    print("Error: Failed to load the image.")
                else:
                    # Get the height and width of the image
                    height, width = image.shape[:2]
                self.maxHeight = height if height > self.maxHeight else self.maxHeight
                self.maxWidth = width if width > self.maxWidth else self.maxWidth

        def __iter__(self):
            return self

        def __next__(self):
            batch = []
            batch_of_numpy = []
            labels = []
            roi_height = []
            roi_width = []
            self.out_image = np.zeros((self.batch_size, self.maxHeight, self.maxWidth, 3), dtype = "uint8")
            for x in range(self.batch_size):
                jpeg_filename = self.files[self.i]
                label = 1
                image = cv2.imread(jpeg_filename, cv2.IMREAD_COLOR)
                # Check if the image was loaded successfully
                if image is None:
                    print("Error: Failed to load the image.")
                else:
                    # Get the height and width of the image
                    height, width = image.shape[:2]
                batch.append(np.asarray(image))
                roi_height.append(height)
                roi_width.append(width)
                self.out_image[x][:roi_height[x],:roi_width[x], :] = batch[x]
                batch_of_numpy.append(self.out_image[x])
                labels.append(1)
                self.i = (self.i + 1) % self.n
            return (batch_of_numpy, labels, roi_height, roi_width, self.maxHeight, self.maxWidth)



# Mode 2
    eii_2 = ExternalInputIteratorMode2(batch_size)

    #Create the pipeline 
    external_source_pipeline_mode2 = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, seed=1, rocal_cpu=True, tensor_layout=types.NCHW)

    with external_source_pipeline_mode2:
        jpegs, labels = fn.external_source(source=eii_2, mode=types.EXTSOURCE_RAW_UNCOMPRESSED, max_width = eii_2.maxWidth, max_height = eii_2.maxHeight)
        output = fn.resize(jpegs, resize_width = 300, resize_height = 300, rocal_tensor_layout = types.NCHW, rocal_tensor_output_type = types.UINT8)
        external_source_pipeline_mode2.set_outputs(output)

    # build the external_source_pipeline_mode2
    external_source_pipeline_mode2.build()
    #Index starting from 0
    cnt = 0
    # Dataloader
    data_loader = ROCALClassificationIterator(external_source_pipeline_mode2, device = "cpu", tensor_dtype = types.UINT8)
    for i, it in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\nImages:\n", it)
            print("**************ends*******************")
            print("**************", i, "*******************")
            for img in it[0]:
                cnt = cnt+1
                draw_patches(img, cnt, device)

if __name__ == '__main__':
    main()