/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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


typedef  struct { float l; float t; float r; float b; } BoundingBoxCord;
typedef  std::vector<BoundingBoxCord> BoundingBoxCords;
typedef  std::vector<int> BoundingBoxLabels;
typedef  struct { int w; int h; } ImgSize;
typedef  std::vector<ImgSize> ImgSizes;

typedef std::vector<std::vector<float>> coords;
typedef std::vector<coords> MaskCords;

struct MetaData
{
    int& get_label() { return _label_id; }
    BoundingBoxCords& get_bb_cords() { return _bb_cords; }
    BoundingBoxLabels& get_bb_labels() { return _bb_label_ids; }
    void set_bb_labels(BoundingBoxLabels bb_label_ids)
    {
        _bb_label_ids = std::move(bb_label_ids);
        _object_count = _bb_label_ids.size();
    }
    ImgSize& get_img_size() {return _img_size; }
    MaskCords& get_mask_cords() { return _mask_cords;}
    int get_object_count() { return _object_count; }
    std::vector<unsigned> get_bb_label_dims()
    {
        _bb_labels_dims = {(unsigned) _bb_label_ids.size()};
        return _bb_labels_dims;
    }
    std::vector<unsigned> get_bb_cords_dims()
    {
        _bb_coords_dims = {(unsigned) _bb_cords.size(), 4};
        return _bb_coords_dims;
    }

protected:
    BoundingBoxCords _bb_cords = {}; // For bb use
    BoundingBoxLabels _bb_label_ids = {};// For bb use
    ImgSize _img_size = {};
    int _label_id = -1; // For label use only
    MaskCords _mask_cords = {};
    std::vector<unsigned> _bb_labels_dims = {};
    std::vector<unsigned> _bb_coords_dims = {};
    std::vector<unsigned> _mask_coords_dims = {};
    int _object_count = 0;
};

struct Label : public MetaData
{
    Label(int label)
    {
        _label_id = label;
        _object_count = 1;
    }
    Label(){ _label_id = -1; }
};

struct BoundingBox : public MetaData
{
    BoundingBox()= default;
    BoundingBox(BoundingBoxCords bb_cords,BoundingBoxLabels bb_label_ids )
    {
        _bb_cords =std::move(bb_cords);
        _bb_label_ids = std::move(bb_label_ids);
    }
    BoundingBox(BoundingBoxCords bb_cords,BoundingBoxLabels bb_label_ids ,ImgSize img_size)
    {
        _bb_cords =std::move(bb_cords);
        _bb_label_ids = std::move(bb_label_ids);
        _img_size = std::move(img_size);
    }
    BoundingBox(BoundingBoxCords bb_cords,BoundingBoxLabels bb_label_ids ,ImgSize img_size, MaskCords mask_cords)
    {
        _bb_cords = std::move(bb_cords);
        _bb_label_ids = std::move(bb_label_ids);
        _img_size = std::move(img_size);
        _mask_cords = std::move(mask_cords);
    }
    void set_bb_cords(BoundingBoxCords bb_cords) { _bb_cords = std::move(bb_cords); }
    void set_bb_labels(BoundingBoxLabels bb_label_ids) { _bb_label_ids = std::move(bb_label_ids); }
    void set_img_sizes(ImgSize img_size) { _img_size =std::move(img_size); }
    void set_mask_cords(MaskCords mask_cords) { _mask_cords = std::move(mask_cords);}
};

struct MetaDataDimensionsBatch
{
    std::vector<std::vector<unsigned>>& bb_labels_dims() { return _bb_labels_dims; }
    std::vector<std::vector<unsigned>>& bb_cords_dims() { return _bb_coords_dims; }
    void clear()
    {
        _bb_labels_dims.clear();
        _bb_coords_dims.clear();
        _mask_coords_dims.clear();
    }
    void resize(unsigned size)
    {
        _bb_labels_dims.resize(size);
        _bb_coords_dims.resize(size);
        _mask_coords_dims.resize(size);
    }
    void insert(MetaDataDimensionsBatch &other)
    {
        _bb_labels_dims.insert(_bb_labels_dims.end(), other.bb_labels_dims().begin(), other.bb_labels_dims().end());
        _bb_coords_dims.insert(_bb_coords_dims.end(), other.bb_cords_dims().begin(), other.bb_cords_dims().end());

    }
private:
    std::vector<std::vector<unsigned>> _bb_labels_dims = {};
    std::vector<std::vector<unsigned>> _bb_coords_dims = {};
    std::vector<std::vector<unsigned>> _mask_coords_dims = {};
};

struct MetaDataBatch
{
    virtual ~MetaDataBatch() = default;
    virtual void clear() = 0;
    virtual void resize(int batch_size) = 0;
    virtual int size() = 0;
    virtual int mask_size() = 0;
    virtual void copy_data(std::vector<void*> buffer) = 0;
    virtual std::vector<unsigned>& get_buffer_size() = 0;
    virtual MetaDataBatch&  operator += (MetaDataBatch& other) = 0;
    MetaDataBatch* concatenate(MetaDataBatch* other)
    {
        *this += *other;
        return this;
    }
    virtual std::shared_ptr<MetaDataBatch> clone()  = 0;
    std::vector<int>& get_label_batch() { return _label_id; }
    std::vector<BoundingBoxCords>& get_bb_cords_batch() { return _bb_cords; }
    std::vector<BoundingBoxLabels>& get_bb_labels_batch() { return _bb_label_ids; }
    ImgSizes& get_img_sizes_batch() { return _img_sizes; }
    std::vector<MaskCords>& get_mask_cords_batch() { return _mask_cords; }
    void reset_objects_count() { _objects_count = 0; }
    void increment_object_count(int count) { _objects_count += count; }
    MetaDataDimensionsBatch& get_metadata_dimensions_batch() { return _metadata_dimensions; }
protected:
    std::vector<int> _label_id = {}; // For label use only
    std::vector<BoundingBoxCords> _bb_cords = {};
    std::vector<BoundingBoxLabels> _bb_label_ids = {};
    ImgSizes _img_sizes = {};
    std::vector<MaskCords> _mask_cords = {};
    std::vector<unsigned> _buffer_size;
    int _objects_count = 0;
    MetaDataDimensionsBatch _metadata_dimensions;
};

struct LabelBatch : public MetaDataBatch
{
    void clear() override
    {
        _label_id.clear();
        _buffer_size.clear();
        _objects_count = 0;
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _label_id.insert(_label_id.end(),other.get_label_batch().begin(), other.get_label_batch().end());
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
    int mask_size() override
    {
        return 0;
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
        mempcpy((int *)buffer[0], _label_id.data(), _label_id.size() * sizeof(int));
    }
    std::vector<unsigned>& get_buffer_size() override
    {
        _buffer_size.emplace_back(_objects_count * sizeof(int));
        return _buffer_size;
    }
};

struct BoundingBoxBatch: public MetaDataBatch
{
    void clear() override
    {
        _bb_cords.clear();
        _bb_label_ids.clear();
        _img_sizes.clear();
        _mask_cords.clear();
        _metadata_dimensions.clear();
        _objects_count = 0;
        _buffer_size.clear();
    }
    MetaDataBatch&  operator += (MetaDataBatch& other) override
    {
        _bb_cords.insert(_bb_cords.end(),other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        _bb_label_ids.insert(_bb_label_ids.end(), other.get_bb_labels_batch().begin(), other.get_bb_labels_batch().end());
        _img_sizes.insert(_img_sizes.end(),other.get_img_sizes_batch().begin(), other.get_img_sizes_batch().end());
        _mask_cords.insert(_mask_cords.end(),other.get_mask_cords_batch().begin(), other.get_mask_cords_batch().end());
        _metadata_dimensions.insert(other.get_metadata_dimensions_batch());
        return *this;
    }
    void resize(int batch_size) override
    {
        _bb_cords.resize(batch_size);
        _bb_label_ids.resize(batch_size);
        _img_sizes.resize(batch_size);
        _mask_cords.resize(batch_size);
        _metadata_dimensions.resize(batch_size);
    }
    int size() override
    {
        return _bb_cords.size();
    }
    int mask_size() override
    {
        return _mask_cords.size();
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
        float *bbox_buffer = (float *)buffer[1];
        auto bb_labels_dims = _metadata_dimensions.bb_labels_dims();
        auto bb_coords_dims = _metadata_dimensions.bb_cords_dims();
        for(unsigned i = 0; i < _bb_label_ids.size(); i++)
        {
            mempcpy(labels_buffer, _bb_label_ids[i].data(), bb_labels_dims[i][0] * sizeof(int));
            memcpy(bbox_buffer, _bb_cords[i].data(), bb_coords_dims[i][0] * sizeof(BoundingBoxCord));
            labels_buffer += bb_labels_dims[i][0];
            bbox_buffer += (bb_coords_dims[i][0] * 4);
        }
    }
    std::vector<unsigned>& get_buffer_size() override
    {
        _buffer_size.emplace_back(_objects_count * sizeof(int));
        _buffer_size.emplace_back(_objects_count * 4 * sizeof(float));
        return _buffer_size;
    }
};
using ImageNameBatch = std::vector<std::string>;
using pMetaData = std::shared_ptr<Label>;
using pMetaDataBox = std::shared_ptr<BoundingBox>;
using pMetaDataBatch = std::shared_ptr<MetaDataBatch>;
