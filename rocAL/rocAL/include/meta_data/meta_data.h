/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "commons.h"


//Defined constants since needed in reader and meta nodes for Pose Estimation
#define NUMBER_OF_JOINTS 17
#define NUMBER_OF_JOINTS_HALFBODY 8
#define PIXEL_STD  200
#define SCALE_CONSTANT_CS 1.25
#define SCALE_CONSTANT_HALF_BODY 1.5
typedef struct BoundingBoxCord_
{
  float l; float t; float r; float b;
  BoundingBoxCord_() {}
  BoundingBoxCord_(float l_, float t_, float r_, float b_): l(l_), t(t_), r(r_), b(b_) {}   // constructor
  BoundingBoxCord_(const BoundingBoxCord_& cord) : l(cord.l), t(cord.t), r(cord.r), b(cord.b) {}  //copy constructor
} BoundingBoxCord;

typedef  struct { float xc; float yc; float w; float h; } BoundingBoxCord_xcycwh;
typedef  std::vector<BoundingBoxCord> BoundingBoxCords;
typedef  std::vector<BoundingBoxCord_xcycwh> BoundingBoxCords_xcycwh;
typedef  std::vector<int> Labels;
typedef  struct { int w; int h; } ImgSize;
typedef  std::vector<ImgSize> ImgSizes;

typedef std::vector<int> ImageIDBatch,AnnotationIDBatch;
typedef std::vector<std::string> ImagePathBatch;
typedef std::vector<float> Joint,JointVisibility,ScoreBatch,RotationBatch;
typedef std::vector<std::vector<float>> Joints,JointsVisibility, CenterBatch, ScaleBatch;
typedef std::vector<std::vector<std::vector<float>>> JointsBatch, JointsVisibilityBatch;

typedef struct
{
    int image_id;
    int annotation_id;
    std::string image_path;
    float center[2];
    float scale[2];
    Joints joints;
    JointsVisibility joints_visibility;
    float score;
    float rotation;
}JointsData;

typedef struct
{
    ImageIDBatch image_id_batch;
    AnnotationIDBatch annotation_id_batch;
    ImagePathBatch image_path_batch;
    CenterBatch center_batch;
    ScaleBatch scale_batch;
    JointsBatch joints_batch;
    JointsVisibilityBatch joints_visibility_batch;
    ScoreBatch score_batch;
    RotationBatch rotation_batch;
}JointsDataBatch;

typedef struct MetaDataInfo {
public:
    uint _img_id = -1;
    std::string _img_name = "";
    ImgSize _img_size = {};
} MetaDataInfo;

struct MetaData
{
    virtual std::vector<int>& get_label() { };
    virtual void set_labels(Labels label_ids) { };
    virtual BoundingBoxCords& get_bb_cords() { };
    virtual BoundingBoxCords_xcycwh& get_bb_cords_xcycwh() { };
    virtual void set_bb_cords_xcycwh(BoundingBoxCords_xcycwh bb_cords_xcycwh) { };
    virtual void set_bb_cords(BoundingBoxCords bb_cords) { };
    ImgSize& get_img_size() {return _info._img_size; }
    std::string& get_image_name() { return _info._img_name; }
    uint& get_image_id() { return _info._img_id; }
    void set_img_size(ImgSize img_size) { _info._img_size = std::move(img_size); }
    void set_img_id(uint img_id) { _info._img_id = std::move(img_id); }
    void set_img_name(std::string img_name) { _info._img_name = std::move(img_name); }
    void set_metadata_info(MetaDataInfo info) { _info = std::move(info); }
    protected:
    MetaDataInfo _info;
};

struct Label : public MetaData
{
    Label(int label)
    {
        _label_ids.resize(1);
        _label_ids[0] = label;
    }
    Label()
    {
        _label_ids.resize(1);
        _label_ids[0] = -1;
    }
    std::vector<int>& get_label() { return _label_ids; }
    void set_labels(Labels label_ids)
    {
        _label_ids = std::move(label_ids);
    }
    protected:
    Labels _label_ids = {}; // For label use only
};

struct BoundingBox : public Label
{
    BoundingBox()= default;
    BoundingBox(BoundingBoxCords bb_cords, Labels bb_label_ids)
    {
        _bb_cords =std::move(bb_cords);
        _label_ids = std::move(bb_label_ids);
    }
    BoundingBox(BoundingBoxCords bb_cords, Labels bb_label_ids, ImgSize img_size, uint img_id)
    {
        _bb_cords =std::move(bb_cords);
        _label_ids = std::move(bb_label_ids);
        _info._img_size = std::move(img_size);
        _info._img_id = std::move(img_id);
    }
    BoundingBox(BoundingBoxCords_xcycwh bb_cords_xcycwh, Labels bb_label_ids)
    {
        _bb_cords_xcycwh =std::move(bb_cords_xcycwh);
        _label_ids = std::move(bb_label_ids);
    }
    BoundingBox(BoundingBoxCords_xcycwh bb_cords_xcycwh, Labels bb_label_ids, ImgSize img_size, uint img_id)
    {
        _bb_cords_xcycwh =std::move(bb_cords_xcycwh);
        _label_ids = std::move(bb_label_ids);
        _info._img_size = std::move(img_size);
        _info._img_id = std::move(img_id);
    }
    BoundingBoxCords& get_bb_cords() { return _bb_cords; }
    BoundingBoxCords_xcycwh& get_bb_cords_xcycwh() { return _bb_cords_xcycwh; }
    void set_bb_cords_xcycwh(BoundingBoxCords_xcycwh bb_cords_xcycwh) { _bb_cords_xcycwh =std::move(bb_cords_xcycwh); }
    void set_bb_cords(BoundingBoxCords bb_cords) { _bb_cords =std::move(bb_cords); }
protected:
    BoundingBoxCords _bb_cords = {}; // For bb use
    BoundingBoxCords_xcycwh _bb_cords_xcycwh = {}; // For bb use
};

struct KeyPoint : public MetaData
{
    KeyPoint()= default;
    KeyPoint(ImgSize img_size, JointsData *joints_data)
    {
        _img_size = std::move(img_size);
        _joints_data = std::move(*joints_data);
    }
    void set_joints_data(JointsData *joints_data) { _joints_data = std::move(*joints_data); }
};

struct MetaDataDimensionsBatch
{
    std::vector<std::vector<size_t>>& labels_dims() { return _labels_dims; }
    std::vector<std::vector<size_t>>& bb_cords_dims() { return _bb_coords_dims; }
    void clear()
    {
        _labels_dims.clear();
        _bb_coords_dims.clear();
    }
    void resize(size_t size)
    {
        _labels_dims.resize(size);
        _bb_coords_dims.resize(size);
    }
    void insert(MetaDataDimensionsBatch &other)
    {
        _labels_dims.insert(_labels_dims.end(), other.labels_dims().begin(), other.labels_dims().end());
        _bb_coords_dims.insert(_bb_coords_dims.end(), other.bb_cords_dims().begin(), other.bb_cords_dims().end());
    }
private:
    std::vector<std::vector<size_t>> _labels_dims = {};
    std::vector<std::vector<size_t>> _bb_coords_dims = {};
};

struct MetaDataInfoBatch {
public:
    std::vector<uint> _img_ids = {};
    std::vector<std::string> _img_names = {};
    std::vector<ImgSize> _img_sizes = {};
};


struct MetaDataBatch
{
    virtual ~MetaDataBatch() = default;
    virtual void clear() = 0;
    virtual void resize(int batch_size) = 0;
    virtual int size() = 0;
    virtual void copy_data(std::vector<void*> buffer) = 0;
    virtual std::vector<size_t>& get_buffer_size() = 0;
    virtual MetaDataBatch&  operator += (MetaDataBatch& other) = 0;
    MetaDataBatch* concatenate(MetaDataBatch* other)
    {
        *this += *other;
        return this;
    }
    virtual std::shared_ptr<MetaDataBatch> clone()  = 0;
    virtual std::vector<Labels>& get_label_batch() { };
    virtual std::vector<BoundingBoxCords>& get_bb_cords_batch() { };
    virtual std::vector<BoundingBoxCords_xcycwh>& get_bb_cords_batch_xcycxwh() { };
    std::vector<uint>& get_image_id_batch() { return _info_batch._img_ids; }
    std::vector<std::string>& get_image_names_batch() {return _info_batch._img_names; }
    ImgSizes& get_img_sizes_batch() { return _info_batch._img_sizes; }
    JointsDataBatch & get_joints_data_batch() { return _joints_data; }
    MetaDataDimensionsBatch& get_metadata_dimensions_batch() { return _metadata_dimensions; }
    int get_batch_object_count() { return _total_objects_count; }
    void reset_objects_count() {
        _total_objects_count = 0;
    }
    void increment_object_count(int count) { _total_objects_count += count; }
protected:
    MetaDataDimensionsBatch _metadata_dimensions;
    MetaDataInfoBatch _info_batch;
    int _total_objects_count = 0;
};

struct LabelBatch : public MetaDataBatch
{
    void clear() override
    {
        for (int i = 0; i < _label_ids.size(); i++) {
            _label_ids[i].clear();
        }
        _label_ids.clear();
        _buffer_size.clear();
        _total_objects_count = 0;
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _label_ids.insert(_label_ids.end(), other.get_label_batch().begin(), other.get_label_batch().end());
        return *this;
    }
    void resize(int batch_size) override
    {
        _label_ids.resize(batch_size);
    }
    int size() override
    {
        return _label_ids.size();
    }
    std::shared_ptr<MetaDataBatch> clone() override
    {
        return std::make_shared<LabelBatch>(*this);
    }
    explicit LabelBatch(std::vector<Labels>& labels)
    {
        _label_ids = std::move(labels);
    }
    LabelBatch() = default;
    void copy_data(std::vector<void*> buffer) override
    {
        if(buffer.size() < 1)
            THROW("The buffers are insufficient") // TODO -change
        auto labels_buffer = (int *)buffer[0];
        for (int i = 0; i < _label_ids.size(); i++) {
            memcpy(labels_buffer, _label_ids[i].data(), _label_ids[i].size() * sizeof(int));
            labels_buffer += _label_ids[i].size();
        }
    }
    std::vector<size_t>& get_buffer_size() override
    {
        _buffer_size.emplace_back(_total_objects_count * sizeof(int));
        return _buffer_size;
    }
    std::vector<Labels>& get_label_batch() { return _label_ids; }
    protected:
    std::vector<Labels> _label_ids = {};
    std::vector<size_t> _buffer_size;
};

struct BoundingBoxBatch: public LabelBatch
{
    void clear() override
    {
        _bb_cords.clear();
        _label_ids.clear();
        _info_batch._img_sizes.clear();
        _info_batch._img_ids.clear();
        _metadata_dimensions.clear();
        _total_objects_count = 0;
        _buffer_size.clear();
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _bb_cords.insert(_bb_cords.end(), other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        _label_ids.insert(_label_ids.end(), other.get_label_batch().begin(), other.get_label_batch().end());
        _info_batch._img_sizes.insert(_info_batch._img_sizes.end(), other.get_img_sizes_batch().begin(), other.get_img_sizes_batch().end());
        _info_batch._img_ids.insert(_info_batch._img_ids.end(), other.get_image_id_batch().begin(), other.get_image_id_batch().end());
        _metadata_dimensions.insert(other.get_metadata_dimensions_batch());
        return *this;
    }
    void resize(int batch_size) override
    {
        _bb_cords.resize(batch_size);
        _label_ids.resize(batch_size);
        _info_batch._img_sizes.resize(batch_size);
        _info_batch._img_ids.resize(batch_size);
        _metadata_dimensions.resize(batch_size);
    }
    int size() override
    {
        return _bb_cords.size();
    }
    std::shared_ptr<MetaDataBatch> clone() override
    {
        return std::make_shared<BoundingBoxBatch>(*this);
    }
    void copy_data(std::vector<void*> buffer) override
    {
        if(buffer.size() < 2)
            THROW("The buffers are insufficient") // TODO -change
        int *labels_buffer = (int *)buffer[0];
        double *bbox_buffer = (double *)buffer[1];
        auto labels_dims = _metadata_dimensions.labels_dims();
        auto bb_coords_dims = _metadata_dimensions.bb_cords_dims();
        for(unsigned i = 0; i < _label_ids.size(); i++)
        {
            memcpy(labels_buffer, _label_ids[i].data(), labels_dims[i][0] * sizeof(int));
            memcpy(bbox_buffer, _bb_cords[i].data(), bb_coords_dims[i][0] * sizeof(BoundingBoxCord));
            labels_buffer += labels_dims[i][0];
            bbox_buffer += (bb_coords_dims[i][0] * 4);
        }
    }
    std::vector<size_t>& get_buffer_size() override
    {
        _buffer_size.emplace_back(_total_objects_count * sizeof(int));
        _buffer_size.emplace_back(_total_objects_count * 4 * sizeof(double));
        return _buffer_size;
    }
    std::vector<BoundingBoxCords>& get_bb_cords_batch() { return _bb_cords; }
    std::vector<BoundingBoxCords_xcycwh>& get_bb_cords_batch_xcycxwh() { return _bb_cords_xcycwh; }

    protected:
    std::vector<BoundingBoxCords> _bb_cords = {};
    std::vector<BoundingBoxCords_xcycwh> _bb_cords_xcycwh = {};
};

struct KeyPointBatch : public MetaDataBatch
{
    void clear() override
    {
        _img_sizes.clear();
        _joints_data = {};
        _bb_cords.clear();
        _bb_label_ids.clear();
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _img_sizes.insert(_img_sizes.end(), other.get_img_sizes_batch().begin(), other.get_img_sizes_batch().end());
        _joints_data.image_id_batch.insert(_joints_data.image_id_batch.end(), other.get_joints_data_batch().image_id_batch.begin(), other.get_joints_data_batch().image_id_batch.end());
        _joints_data.annotation_id_batch.insert(_joints_data.annotation_id_batch.end(), other.get_joints_data_batch().annotation_id_batch.begin(), other.get_joints_data_batch().annotation_id_batch.end());
        _joints_data.center_batch.insert(_joints_data.center_batch.end(), other.get_joints_data_batch().center_batch.begin(), other.get_joints_data_batch().center_batch.end());
        _joints_data.scale_batch.insert(_joints_data.scale_batch.end(), other.get_joints_data_batch().scale_batch.begin(), other.get_joints_data_batch().scale_batch.end());
        _joints_data.joints_batch.insert(_joints_data.joints_batch.end(), other.get_joints_data_batch().joints_batch.begin() ,other.get_joints_data_batch().joints_batch.end());
        _joints_data.joints_visibility_batch.insert(_joints_data.joints_visibility_batch.end(), other.get_joints_data_batch().joints_visibility_batch.begin(), other.get_joints_data_batch().joints_visibility_batch.end());
        _joints_data.score_batch.insert(_joints_data.score_batch.end(), other.get_joints_data_batch().score_batch.begin(), other.get_joints_data_batch().score_batch.end());
        _joints_data.rotation_batch.insert(_joints_data.rotation_batch.end(), other.get_joints_data_batch().rotation_batch.begin(), other.get_joints_data_batch().rotation_batch.end());
        return *this;
    }
    void resize(int batch_size) override
    {
        _joints_data.image_id_batch.resize(batch_size);
        _joints_data.annotation_id_batch.resize(batch_size);
        _joints_data.center_batch.resize(batch_size);
        _joints_data.scale_batch.resize(batch_size);
        _joints_data.joints_batch.resize(batch_size);
        _joints_data.joints_visibility_batch.resize(batch_size);
        _joints_data.score_batch.resize(batch_size);
        _joints_data.rotation_batch.resize(batch_size);
        _bb_cords.resize(batch_size);
        _bb_label_ids.resize(batch_size);
    }
    int size() override
    {
        return _joints_data.image_id_batch.size();
    }
    std::shared_ptr<MetaDataBatch> clone() override
    {
        return std::make_shared<KeyPointBatch>(*this);
    }
    void copy_data(std::vector<void*> buffer) override {}
    std::vector<size_t>& get_buffer_size() override { return _buffer_size; }
};

using ImageNameBatch = std::vector<std::string>;
using pMetaData = std::shared_ptr<Label>;
using pMetaDataBox = std::shared_ptr<BoundingBox>;
using pMetaDataKeyPoint = std::shared_ptr<KeyPoint>;
using pMetaDataBatch = std::shared_ptr<MetaDataBatch>;

