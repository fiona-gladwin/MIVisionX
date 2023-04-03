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
#include <cstring>
#include "commons.h"

#define ANCHOR_SIZE 120087

typedef struct BoundingBoxCord_
{
  double l; double t; double r; double b;
  BoundingBoxCord_() {}
  BoundingBoxCord_(double l_, double t_, double r_, double b_): l(l_), t(t_), r(r_), b(b_) {}   // constructor
  BoundingBoxCord_(const BoundingBoxCord_& cord) : l(cord.l), t(cord.t), r(cord.r), b(cord.b) {}  //copy constructor
} BoundingBoxCord;

typedef  struct { double xc; double yc; double w; double h; } BoundingBoxCord_xcycwh;
typedef  std::vector<BoundingBoxCord> BoundingBoxCords;
typedef  std::vector<BoundingBoxCord_xcycwh> BoundingBoxCords_xcycwh;
typedef  std::vector<int> BoundingBoxLabels;
typedef  struct { int w; int h; } ImgSize;
typedef  std::vector<ImgSize> ImgSizes;

typedef struct MetaDataInfo {
public:
    uint _img_id = -1;
    std::string _img_name = "";
    ImgSize _img_size = {};
} MetaDataInfo;

struct MetaData
{
    virtual int& get_label() { };
    virtual BoundingBoxCords& get_bb_cords() { };
    virtual BoundingBoxCords_xcycwh& get_bb_cords_xcycwh() { };
    virtual BoundingBoxLabels& get_bb_labels() { };
    virtual int get_object_count() { };
    virtual std::vector<size_t> get_bb_label_dims() { };
    virtual std::vector<size_t> get_bb_cords_dims() { };
    virtual void set_bb_labels(BoundingBoxLabels bb_label_ids) { };
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
        _label_id = label;
        _object_count = 1;
    }
    Label(){ _label_id = -1; }
    int& get_label() { return _label_id; }
    int get_object_count() { return _object_count; }
    protected:
    int _label_id = -1; // For label use only
    int _object_count = 0;
};

struct BoundingBox : public MetaData
{
    BoundingBox()= default;
    BoundingBox(BoundingBoxCords bb_cords, BoundingBoxLabels bb_label_ids)
    {
        _bb_cords =std::move(bb_cords);
        _bb_label_ids = std::move(bb_label_ids);
    }
    BoundingBox(BoundingBoxCords bb_cords, BoundingBoxLabels bb_label_ids, ImgSize img_size, uint img_id)
    {
        _bb_cords =std::move(bb_cords);
        _bb_label_ids = std::move(bb_label_ids);
        _info._img_size = std::move(img_size);
        _info._img_id = std::move(img_id);
    }
    BoundingBox(BoundingBoxCords_xcycwh bb_cords_xcycwh, BoundingBoxLabels bb_label_ids)
    {
        _bb_cords_xcycwh =std::move(bb_cords_xcycwh);
        _bb_label_ids = std::move(bb_label_ids);
    }
    BoundingBox(BoundingBoxCords_xcycwh bb_cords_xcycwh, BoundingBoxLabels bb_label_ids, ImgSize img_size, uint img_id)
    {
        _bb_cords_xcycwh =std::move(bb_cords_xcycwh);
        _bb_label_ids = std::move(bb_label_ids);
        _info._img_size = std::move(img_size);
        _info._img_id = std::move(img_id);
    }
    BoundingBoxCords& get_bb_cords() { return _bb_cords; }
    BoundingBoxCords_xcycwh& get_bb_cords_xcycwh() { return _bb_cords_xcycwh; }
    BoundingBoxLabels& get_bb_labels() { return _bb_label_ids; }
    int get_object_count() { return _object_count; }
    std::vector<size_t> get_bb_label_dims()
    {
        _bb_labels_dims = {_bb_label_ids.size()};
        return _bb_labels_dims;
    }
    std::vector<size_t> get_bb_cords_dims()
    {
        _bb_coords_dims = {_bb_cords.size(), 4};
        return _bb_coords_dims;
    }
    void set_bb_labels(BoundingBoxLabels bb_label_ids)
    {
        _bb_label_ids = std::move(bb_label_ids);
        _object_count = _bb_label_ids.size();
    }
    void set_bb_cords_xcycwh(BoundingBoxCords_xcycwh bb_cords_xcycwh) { _bb_cords_xcycwh =std::move(bb_cords_xcycwh); }
    void set_bb_cords(BoundingBoxCords bb_cords) { _bb_cords =std::move(bb_cords); }
protected:
    BoundingBoxCords _bb_cords = {}; // For bb use
    BoundingBoxCords_xcycwh _bb_cords_xcycwh = {}; // For bb use
    BoundingBoxLabels _bb_label_ids = {};// For bb use
    std::vector<size_t> _bb_labels_dims = {};
    std::vector<size_t> _bb_coords_dims = {};
    int _object_count = 0;
};

struct MetaDataDimensionsBatch
{
    std::vector<std::vector<size_t>>& bb_labels_dims() { return _bb_labels_dims; }
    std::vector<std::vector<size_t>>& bb_cords_dims() { return _bb_coords_dims; }
    void clear()
    {
        _bb_labels_dims.clear();
        _bb_coords_dims.clear();
    }
    void resize(size_t size)
    {
        _bb_labels_dims.resize(size);
        _bb_coords_dims.resize(size);
    }
    void insert(MetaDataDimensionsBatch &other)
    {
        _bb_labels_dims.insert(_bb_labels_dims.end(), other.bb_labels_dims().begin(), other.bb_labels_dims().end());
        _bb_coords_dims.insert(_bb_coords_dims.end(), other.bb_cords_dims().begin(), other.bb_cords_dims().end());
    }
private:
    std::vector<std::vector<size_t>> _bb_labels_dims = {};
    std::vector<std::vector<size_t>> _bb_coords_dims = {};
};

struct MetaDataInfoBatch {
public:
    std::vector<uint> _img_ids = {};
    std::vector<std::string> _img_names = {};
    std::vector<ImgSize> _img_sizes = {};
};


struct MetaDataBatch : public MetaDataInfoBatch
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
    virtual std::vector<int>& get_label_batch() { };
    virtual std::vector<BoundingBoxCords>& get_bb_cords_batch() { };
    virtual std::vector<BoundingBoxCords_xcycwh>& get_bb_cords_batch_xcycxwh() { };
    virtual std::vector<BoundingBoxLabels>& get_bb_labels_batch() { };
    std::vector<uint>& get_image_id_batch() { return _info_batch._img_ids; }
    std::vector<std::string>& get_image_names_batch() {return _info_batch._img_names; }
    ImgSizes& get_img_sizes_batch() { return _info_batch._img_sizes; }
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
        _label_id.clear();
        _buffer_size.clear();
        _total_objects_count = 0;
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _label_id.insert(_label_id.end(), other.get_label_batch().begin(), other.get_label_batch().end());
        return *this;
    }
    void resize(int batch_size) override
    {
        _label_id.resize(batch_size);
    }
    int size() override
    {
        return _label_id.size();
    }
    std::shared_ptr<MetaDataBatch> clone() override
    {
        return std::make_shared<LabelBatch>(*this);
    }
    explicit LabelBatch(std::vector<int>& labels)
    {
        _label_id = std::move(labels);
    }
    LabelBatch() = default;
    void copy_data(std::vector<void*> buffer) override
    {
        if(buffer.size() < 1)
            THROW("The buffers are insufficient") // TODO -change
        memcpy((int *)buffer[0], _label_id.data(), _label_id.size() * sizeof(int));
    }
    std::vector<size_t>& get_buffer_size() override
    {
        _buffer_size.emplace_back(_total_objects_count * sizeof(int));
        return _buffer_size;
    }
    std::vector<int>& get_label_batch() { return _label_id; }
    protected:
    std::vector<int> _label_id = {};
    std::vector<size_t> _buffer_size;
};

struct BoundingBoxBatch: public MetaDataBatch
{
    void clear() override
    {
        _bb_cords.clear();
        _bb_label_ids.clear();
        _info_batch._img_sizes.clear();
        _info_batch._img_ids.clear();
        _metadata_dimensions.clear();
        _total_objects_count = 0;
        _buffer_size.clear();
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _bb_cords.insert(_bb_cords.end(), other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        _bb_label_ids.insert(_bb_label_ids.end(), other.get_bb_labels_batch().begin(), other.get_bb_labels_batch().end());
        _info_batch._img_sizes.insert(_info_batch._img_sizes.end(), other.get_img_sizes_batch().begin(), other.get_img_sizes_batch().end());
        _info_batch._img_ids.insert(_info_batch._img_ids.end(), other.get_image_id_batch().begin(), other.get_image_id_batch().end());
        _metadata_dimensions.insert(other.get_metadata_dimensions_batch());
        return *this;
    }
    void resize(int batch_size) override
    {
        _bb_cords.resize(batch_size);
        _bb_label_ids.resize(batch_size);
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
        auto bb_labels_dims = _metadata_dimensions.bb_labels_dims();
        auto bb_coords_dims = _metadata_dimensions.bb_cords_dims();
        for(unsigned i = 0; i < _bb_label_ids.size(); i++)
        {
            memcpy(labels_buffer, _bb_label_ids[i].data(), bb_labels_dims[i][0] * sizeof(int));
            memcpy(bbox_buffer, _bb_cords[i].data(), bb_coords_dims[i][0] * sizeof(BoundingBoxCord));
            labels_buffer += bb_labels_dims[i][0];
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
    std::vector<BoundingBoxLabels>& get_bb_labels_batch() { return _bb_label_ids; }
    protected:
    std::vector<BoundingBoxCords> _bb_cords = {};
    std::vector<BoundingBoxCords_xcycwh> _bb_cords_xcycwh = {};
    std::vector<BoundingBoxLabels> _bb_label_ids = {};
    std::vector<size_t> _buffer_size;
};

using ImageNameBatch = std::vector<std::string>;
using pMetaData = std::shared_ptr<Label>;
using pMetaDataBox = std::shared_ptr<BoundingBox>;
using pMetaDataBatch = std::shared_ptr<MetaDataBatch>;

