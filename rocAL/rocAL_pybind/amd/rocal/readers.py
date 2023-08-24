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

##
# @file readers.py
#
# @brief File containing reader functions for multiple datasets and data formats
# File containing reader functions for multiple datasets and data formats

import rocal_pybind as b
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types

def coco(annotations_file='', ltrb=True, masks=False, ratio=False, avoid_class_remapping=False,
         pixelwise_masks=False, is_box_encoder=False, is_box_iou_matcher=False, stick_to_shard=False, pad_last_batch=False):
    """!Creates a COCOReader node.

        @param annotations_file         Path to the COCO annotations file.
        @param ltrb                     Whether bounding box coordinates are provided in (left, top, right, bottom) format.
        @param masks                    Whether masks are included in the annotations.
        @param ratio                    Whether bounding box coordinates are provided in ratio format.
        @param avoid_class_remapping    Specifies if class remapping should be avoided.
        @param pixelwise_masks          Whether pixel-wise masks are included in the annotations.
        @param is_box_encoder           Whether it's used as a box encoder.
        @param is_box_iou_matcher       Whether it's used as a box IoU matcher.
        @param stick_to_shard           Specifies if the reader should stick to a single shard.
        @param pad_last_batch           Specifies if the last batch should be padded.

        @return    meta data, labels, and bounding boxes.
    """
    Pipeline._current_pipeline._reader = "COCOReader"
    # Output
    labels = []
    bboxes = []
    kwargs_pybind = {
        "source_path": annotations_file,
        "is_output": True,
        "mask": masks,
        "ltrb": ltrb,
        "is_box_encoder": is_box_encoder}
    meta_data = b.cocoReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (meta_data, labels, bboxes)

def file(file_root, file_filters=None, file_list='', stick_to_shard=False, pad_last_batch=False):
    """!Creates a labelReader node for loading label data from files.

        @param file_root         Root directory containing label files.
        @param file_filters      Filters to apply to the label files.
        @param file_list         List of label files.
        @param stick_to_shard    Specifies if the reader should stick to a single shard.
        @param pad_last_batch    Specifies if the last batch should be padded.

        @return    label reader meta data and labels.
    """
    Pipeline._current_pipeline._reader = "labelReader"
    # Output
    labels = []
    kwargs_pybind = {"source_path": file_root}
    label_reader_meta_data = b.labelReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (label_reader_meta_data, labels)

def tfrecord(path, user_feature_key_map, features, reader_type=0, stick_to_shard=False, pad_last_batch=False):
    """!Creates a TFRecordReader node for loading TFRecord dataset.

        @param path                    Path to the TFRecord dataset.
        @param user_feature_key_map    User-provided feature key mapping.
        @param features                Features to load from TFRecords.
        @param reader_type             Type of reader (0 for classification, 1 for detection).
        @param stick_to_shard          Specifies if the reader should use only a single shard.
        @param pad_last_batch          Specifies if the last batch should be padded.

        @return    Features loaded from TFRecords.
    """
    labels = []
    if reader_type == 1:
        Pipeline._current_pipeline._reader = "TFRecordReaderDetection"
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map["image/class/label"],
                         "user_key_for_text": user_feature_key_map["image/class/text"], "user_key_for_xmin": user_feature_key_map["image/object/bbox/xmin"],
                         "user_key_for_ymin": user_feature_key_map["image/object/bbox/ymin"], "user_key_for_xmax": user_feature_key_map["image/object/bbox/xmax"],
                         "user_key_for_ymax": user_feature_key_map["image/object/bbox/ymax"], "user_key_for_filename": user_feature_key_map["image/filename"]}
        for key in features.keys():
            if key not in user_feature_key_map.keys():
                print(
                    "For Object Detection, ROCAL TFRecordReader needs all the following keys in the featureKeyMap:")
                print("image/encoded\nimage/class/label\nimage/class/text\nimage/object/bbox/xmin\nimage/object/bbox/ymin\nimage/object/bbox/xmax\nimage/object/bbox/ymax\n")
                exit()
        tf_meta_data = b.tfReaderDetection(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        Pipeline._current_pipeline._reader = "TFRecordReaderClassification"
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map["image/class/label"],
                         "user_key_for_filename": user_feature_key_map["image/filename"]}
        for key in features.keys():
            if key not in user_feature_key_map.keys():
                print(
                    "For Image Classification, ROCAL TFRecordReader needs all the following keys in the featureKeyMap:")
                print("image/encoded\nimage/class/label\n")
                exit()
        tf_meta_data = b.tfReader(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    features["image/encoded"] = tf_meta_data
    features["image/class/label"] = labels
    return features

def caffe(path, bbox=False, stick_to_shard=False, pad_last_batch=False):
    """!Creates a CaffeReader node for loading Caffe dataset.

        @param path              Path to the Caffe dataset.
        @param bbox              Specifies if bounding boxes are included in the dataset.
        @param stick_to_shard    Specifies if the reader should use only single shard.
        @param pad_last_batch    Specifies if the last batch should be padded.

        @return    caffe reader meta data, bboxes, and labels.
    """
    # Output
    bboxes = []
    labels = []
    kwargs_pybind = {"source_path": path}
    # Node Object
    if (bbox == True):
        Pipeline._current_pipeline._reader = "CaffeReaderDetection"
        caffe_reader_meta_data = b.caffeReaderDetection(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        Pipeline._current_pipeline._reader = "CaffeReader"
        caffe_reader_meta_data = b.caffeReader(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    if (bbox == True):
        return (caffe_reader_meta_data, bboxes, labels)
    else:
        return (caffe_reader_meta_data, labels)

def caffe2(path, bbox=False, stick_to_shard=False, pad_last_batch=False):
    """!Creates a Caffe2Reader node for loading Caffe2 dataset.

    Args:
        @param path              Path to the Caffe2 dataset.
        @param bbox              Specifies if bounding boxes are included in the dataset.
        @param stick_to_shard    Specifies if the reader should stick to a single shard.
        @param pad_last_batch    Specifies if the last batch should be padded.

        @return    caffe2 reader meta data, bboxes, and labels.
    """
    # Output
    bboxes = []
    labels = []
    kwargs_pybind = {"source_path": path, "is_output": True}
    if (bbox == True):
        Pipeline._current_pipeline._reader = "Caffe2ReaderDetection"
        caffe2_meta_data = b.caffe2ReaderDetection(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        Pipeline._current_pipeline._reader = "Caffe2Reader"
        caffe2_meta_data = b.caffe2Reader(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    if (bbox == True):
        return (caffe2_meta_data, bboxes, labels)
    else:
        return (caffe2_meta_data, labels)

def video(sequence_length, file_list_frame_num=False, file_root="", image_type=types.RGB, num_shards=1,
          random_shuffle=False, step=1, stride=1, decoder_mode=types.SOFTWARE_DECODE, enable_frame_num=False,
          enable_timestamps=False, file_list="", stick_to_shard=False, pad_last_batch=False,
          file_list_include_preceding_frame=False, normalized=False, skip_vfr_check=False):
    """!Creates a VideoDecoder node for loading video sequences.

    Args:
        @param sequence_length                      Number of frames in video sequence.
        @param file_list_frame_num                  Specifies whether file list includes frame numbers.
        @param file_root                            Root directory containing video files.
        @param image_type                           Color format of the frames.
        @param num_shards                           Number of shards for data parallelism.
        @param random_shuffle                       Specifies if frames should be randomly shuffled.
        @param step                                 Frame step size.
        @param stride                               Frame stride size.
        @param decoder_mode                         Mode of video decoding.
        @param enable_frame_num                     Specifies whether frame numbers are enabled.
        @param enable_timestamps                    Specifies whether timestamps are enabled.
        @param file_list                            List of video files.
        @param stick_to_shard                       Specifies if the reader should stick to a single shard.
        @param pad_last_batch                       Specifies if the last batch should be padded.
        @param file_list_include_preceding_frame    Specifies if file list includes preceding frames.
        @param normalized                           Specifies if video frames should be normalized.
        @param skip_vfr_check                       Specifies whether to skip variable frame rate check.

        @return   list of loaded video sequences.
    """
    Pipeline._current_pipeline._reader = "VideoDecoder"
    # Output
    videos = []
    kwargs_pybind_reader = {
        "source_path": file_root,
        "sequence_length": sequence_length,
        "frame_step": step,
        "frame_stride": stride,
        "file_list_frame_num": file_list_frame_num}  # VideoMetaDataReader
    b.videoMetaDataReader(Pipeline._current_pipeline._handle,
                          *(kwargs_pybind_reader.values()))

    kwargs_pybind_decoder = {
        "source_path": file_root,
        "color_format": image_type,
        "decoder_mode": decoder_mode,
        "shard_count": num_shards,
        "sequence_length": sequence_length,
        "shuffle": random_shuffle,
        "is_output": False,
        "loop": False,
        "frame_step": step,
        "frame_stride": stride,
        "file_list_frame_num": file_list_frame_num}  # VideoDecoder
    videos = b.videoDecoder(
        Pipeline._current_pipeline._handle, *(kwargs_pybind_decoder.values()))
    return (videos)

def video_resize(sequence_length, resize_width, resize_height, file_list_frame_num=False,
                 file_root="", image_type=types.RGB,
                 num_shards=1, random_shuffle=False, step=3,
                 stride=3, decoder_mode=types.SOFTWARE_DECODE,
                 scaling_mode=types.SCALING_MODE_DEFAULT, interpolation_type=types.LINEAR_INTERPOLATION,
                 resize_longer=0, resize_shorter=0, max_size=[], enable_frame_num=False,
                 enable_timestamps=False, file_list="", stick_to_shard=False, pad_last_batch=False,
                 file_list_include_preceding_frame=False, normalized=False, skip_vfr_check=False):
    """!Creates a VideoDecoderResize node in the pipeline for loading and resizing video sequences.

        @param sequence_length                      Number of frames in video sequence.
        @param resize_width                         output width for resizing.
        @param resize_height                        output height for resizing.
        @param file_list_frame_num                  Specifies whether file list includes frame numbers.
        @param file_root                            Root directory containing video files.
        @param image_type                           Color format of the frames.
        @param num_shards                           Number of shards for data parallelism.
        @param random_shuffle                       Specifies if frames should be randomly shuffled.
        @param step                                 Frame step size.
        @param stride                               Frame stride size.
        @param decoder_mode                         Mode of video decoding.
        @param scaling_mode                         Scaling mode for resizing.
        @param interpolation_type                   Interpolation type for resizing.
        @param resize_longer                        Target size for the longer dimension during resizing.
        @param resize_shorter                       Target size for the shorter dimension during resizing.
        @param max_size                             Maximum size for resizing.
        @param enable_frame_num                     Specifies whether frame numbers are enabled.
        @param enable_timestamps                    Specifies whether timestamps are enabled.
        @param file_list                            List of video files.
        @param stick_to_shard                       Specifies if the reader should stick to a single shard.
        @param pad_last_batch                       Specifies if the last batch should be padded.
        @param file_list_include_preceding_frame    Specifies if file list includes preceding frames.
        @param normalized                           Specifies if video frames should be normalized.
        @param skip_vfr_check                       Specifies whether to skip variable frame rate check.

        @returns   loaded and resized video sequences and meta data.
    """
    Pipeline._current_pipeline._reader = "VideoDecoderResize"
    # Output
    videos = []
    kwargs_pybind_reader = {
        "source_path": file_root,
        "sequence_length": sequence_length,
        "frame_step": step,
        "frame_stride": stride,
        "file_list_frame_num": file_list_frame_num}  # VideoMetaDataReader
    meta_data = b.videoMetaDataReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind_reader.values()))

    kwargs_pybind_decoder = {"source_path": file_root, "color_format": image_type, "decoder_mode": decoder_mode, "shard_count": num_shards,
                             "sequence_length": sequence_length, "resize_width": resize_width, "resize_height": resize_height,
                             "shuffle": random_shuffle, "is_output": False, "loop": False, "frame_step": step, "frame_stride": stride,
                             "file_list_frame_num": file_list_frame_num, "scaling_mode": scaling_mode, "max_size": max_size,
                             "resize_shorter": resize_shorter, "resize_longer": resize_longer, "interpolation_type": interpolation_type}
    videos = b.videoDecoderResize(
        Pipeline._current_pipeline._handle, *(kwargs_pybind_decoder.values()))
    return (videos, meta_data)

def sequence_reader(file_root, sequence_length, image_type=types.RGB, num_shards=1, random_shuffle=False, step=3, stride=1, stick_to_shard=False, pad_last_batch=False):
    """!Creates a SequenceReader node for loading image sequences.

        @param file_root            Root directory containing image sequences.
        @param sequence_length      Number of frames in each sequence.
        @param image_type           Color format of the frames.
        @param num_shards           Number of shards for data parallelism.
        @param random_shuffle       Specifies if frames should be randomly shuffled.
        @param step                 Frame step size.
        @param stride               Frame stride size.
        @param stick_to_shard       Specifies if the reader should stick to a single shard.
        @param pad_last_batch       Specifies if the last batch should be padded.

        @return    list of loaded image sequences.
    """
    Pipeline._current_pipeline._reader = "SequenceReader"
    # Output
    kwargs_pybind = {
        "source_path": file_root,
        "color_format": image_type,
        "shard_count": num_shards,
        "sequence_length": sequence_length,
        "is_output": False,
        "shuffle": random_shuffle,
        "loop": False,
        "frame_step": step,
        "frame_stride": stride}
    frames = b.sequenceReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (frames)

def mxnet(path, stick_to_shard=False, pad_last_batch=False):
    """!Creates an MXNETReader node for loading data from MXNet record files.

        @param path              Path to the MXNet record file.
        @param stick_to_shard    Specifies if the reader should stick to a single shard.
        @param pad_last_batch    Specifies if the last batch should be padded.

        @return    Metadata and loaded data from the MXNet record file.
    """
    Pipeline._current_pipeline._reader = "MXNETReader"
    # Output
    kwargs_pybind = {
        "source_path": path,
        "is_output": True
    }
    mxnet_metadata = b.mxnetReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return mxnet_metadata
