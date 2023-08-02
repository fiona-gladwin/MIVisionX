import types
import collections
import numpy as np
from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import amd.rocal.fn as fn
import amd.rocal.types as types

def main():

    batch_size = 5
    data_dir = "/media/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/" # Pass a directory
    device = "cpu"
    def draw_patches(img, idx, device):
    #image is expected as a tensor, bboxes as numpy
        import cv2
        image = img.detach().numpy()
        image = image.transpose([1, 2, 0]) # NCHW
        image = (image).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(idx)+"_"+"train"+".png", image)

    #Define the Data Source for all image samples
    class ExternalInputIteratorMode1(object):
        def __init__(self, batch_size):
            self.images_dir = data_dir
            self.batch_size = batch_size
            self.files = []
            import os, glob
            for filename in glob.glob(os.path.join(self.images_dir, '*.jpg')):
                self.files.append(filename)

        def __iter__(self):
            self.i = 0
            self.n = len(self.files)
            self.maxWidth = None
            self.maxHeight = None
            return self

        def __next__(self):
            batch = []
            labels = []
            srcsize_height = []
            for x in range(self.batch_size):
                jpeg_filename = self.files[self.i]
                label = 1
                f = open(jpeg_filename, 'rb')
                numpy_buffer = np.frombuffer(f.read(), dtype = np.uint8)
                batch.append(numpy_buffer)
                srcsize_height.append(len(numpy_buffer))
                labels.append(1)
                self.i = (self.i + 1) % self.n
            return (batch, labels, srcsize_height)

# Mode 1
    eii_1 = ExternalInputIteratorMode1(batch_size)

    #Create the pipeline 
    external_source_pipeline_mode1 = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, seed=1, rocal_cpu=True, tensor_layout=types.NCHW)

    with external_source_pipeline_mode1:
        jpegs, labels = fn.external_source(source=eii_1, mode=types.EXTSOURCE_RAW_COMPRESSED, max_width=2000, max_height=2000)
        output = fn.resize(jpegs, resize_width=2000, resize_height=2000, rocal_tensor_layout = types.NCHW, rocal_tensor_output_type = types.UINT8)
        external_source_pipeline_mode1.set_outputs(output)

    # build the external_source_pipeline_mode1
    external_source_pipeline_mode1.build()
    #Index starting from 0
    cnt = 0
    # Dataloader
    data_loader = ROCALClassificationIterator(external_source_pipeline_mode1,device="cpu", tensor_dtype = types.UINT8)
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
