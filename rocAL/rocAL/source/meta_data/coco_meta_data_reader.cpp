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

#include "coco_meta_data_reader.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <fstream>
#include <set>
#include "lookahead_parser.h"
#include "maskApi.h"

using namespace std;

void COCOMetaDataReader::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch)
{
    _path = cfg.path();
    _output = meta_data_batch;
    _output->set_metadata_type(cfg.type());
    _max_width = 0;
    _max_height = 0;
}

bool COCOMetaDataReader::exists(const std::string &image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void COCOMetaDataReader::lookup(const std::vector<std::string> &image_names)
{

    if (image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for (unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_labels_batch()[i] = it->second->get_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_size();
        _output->get_image_id_batch()[i] = it->second->get_image_id();
        if (_output->get_metadata_type() == MetaDataType::PolygonMask)
        {
            auto mask_cords = it->second->get_mask_cords();
            _output->get_mask_cords_batch()[i] = mask_cords;
            _output->get_mask_polygons_count_batch()[i] = it->second->get_polygon_count();
            _output->get_mask_vertices_count_batch()[i] = it->second->get_vertices_count();
        }
        if (_output->get_metadata_type() == MetaDataType::PixelwiseMask)
        {
            _output->get_pixelwise_labels_batch()[i] = it->second->get_pixelwise_label();
        }
    }
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size, MaskCords mask_cords, std::vector<int> polygon_count, std::vector<std::vector<int>> vertices_count)
{
    if (exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        it->second->get_mask_cords().insert(it->second->get_mask_cords().end(), mask_cords.begin(), mask_cords.end());
        it->second->get_polygon_count().push_back(polygon_count[0]);
        it->second->get_vertices_count().push_back(vertices_count[0]);
        return;
    }
    if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
        pMetaDataPolygonMask info = std::make_shared<PolygonMask>(bb_coords, bb_labels, image_size, mask_cords, polygon_count, vertices_count);
        _map_content.insert(pair<std::string, std::shared_ptr<PolygonMask>>(image_name, info));
    } else if (_output->get_metadata_type() == MetaDataType::PixelwiseMask) {
        pMetaDataPixelwiseMask info = std::make_shared<PixelwiseMask>(bb_coords, bb_labels, image_size, mask_cords, polygon_count, vertices_count);
        _map_content.insert(pair<std::string, std::shared_ptr<PixelwiseMask>>(image_name, info));    
    }
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size, int image_id)
{
    if (exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size, image_id);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void COCOMetaDataReader::print_map_contents()
{
    // BoundingBoxCords bb_coords;
    // Labels bb_labels;
    // ImgSize img_size;
    // MaskCords mask_cords;
    // std::vector<int> polygon_size;
    // std::vector<std::vector<int>> vertices_count;

    std::cout << "\nBBox Annotations List: \n";
    for (auto &elem : _map_content)
    {
        std::cout << "\nName :\t " << elem.first;
        auto bb_coords = elem.second->get_bb_cords();
        auto bb_labels = elem.second->get_labels();
        auto img_size = elem.second->get_img_size();
        std::cout << "<wxh, num of bboxes>: " << img_size.w << " X " << img_size.h << " , " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++)
        {
            std::cout << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << "Label Id : " << bb_labels[i] << std::endl;
        }
        if (_output->get_metadata_type() == MetaDataType::PolygonMask)
        {
            int count = 0;
            auto mask_cords = elem.second->get_mask_cords();
            auto polygon_size = elem.second->get_polygon_count();
            auto vertices_count = elem.second->get_vertices_count();
            std::cout << "\nNumber of objects : " << bb_coords.size() << std::endl;
            for (unsigned int i = 0; i < bb_coords.size(); i++)
            {
                std::cout << "\nNumber of polygons for object[ << " << i << "]:" << polygon_size[i];
                for (int j = 0; j < polygon_size[i]; j++)
                {
                    std::cout << "\nPolygon size :" << vertices_count[i][j] << "Elements::";
                    for (int k = 0; k < vertices_count[i][j]; k++, count++)
                        std::cout << "\t " << mask_cords[count];
                }
            }
        }
    }
}

void COCOMetaDataReader::generate_pixelwise_mask(std::string filename, RLE *rle_in) {
    //std::cout  << "generate_pixelwise_mask" << std::endl;
    Labels bb_labels;
    ImgSize img_size;
    MaskCords mask_cords;
    std::vector<int> polygon_size;
    std::vector<std::vector<int>> vertices_count;
    std::map<int, std::vector<RLE> > FromPoly;
    auto it = _map_content.find(filename);
    BoundingBoxCords bb_coords = it->second->get_bb_cords();
    auto &pixelwise_labels = it->second->get_pixelwise_label();
    bb_labels = it->second->get_labels();
    img_size = it->second->get_img_size();
    mask_cords = it->second->get_mask_cords();
    polygon_size = it->second->get_polygon_count();
    vertices_count = it->second->get_vertices_count();
    int h = img_size.h;
    int w = img_size.w;
    pixelwise_labels.resize(h*w);

    if (rle_in) {
        for (unsigned int i = 0; i < bb_coords.size(); i++) {
            bb_labels[i] = _label_info.find(bb_labels[i])->second;
        }
    }
    
    // Generate FromPoly for all polygons in image
    int count = 0;
    for (unsigned int i = 0; i < bb_coords.size(); i++)
    {
        for (int j = 0; j < polygon_size[i]; j++)
        {
            std::vector<double> in;
            for (int k = 0; k < vertices_count[i][j]; k++, count++) {
                in.push_back(mask_cords[count]);
            }
            auto label = bb_labels[i];
            RLE M;
            rleInit(&M, 0, 0, 0, 0);
            
            rleFrPoly(&M, in.data(), in.size() / 2, img_size.h, img_size.w);
            FromPoly[label].push_back(M);
        }
    }

    std::set<int> labels(bb_labels.data(), bb_labels.data() + bb_labels.size());
    if (!labels.size()) {
        return;
    }

    RLE* r_out;
    rlesInit(&r_out, *labels.rbegin() + 1);

    if (rle_in) {
        const auto &rle = rle_in;
        auto mask_idx = bb_labels.size();
        int label = bb_labels[mask_idx];
        rleInit(&r_out[label], rle->h, rle->w, rle->m, rle->cnts);
    }

    for (const auto &rles : FromPoly)
        rleMerge(rles.second.data(), &r_out[rles.first], rles.second.size(), 0);

    struct Encoding {
        uint m;
        std::unique_ptr<uint[]> cnts;
        std::unique_ptr<int[]> vals;
    };
    Encoding A;
    A.cnts = std::make_unique<uint[]>(h * w + 1);  // upper-bound
    A.vals = std::make_unique<int[]>(h * w + 1);

    // first copy the content of the first label to the output
    bool v = false;
    A.m = r_out[*labels.begin()].m;
    for (siz a = 0; a < r_out[*labels.begin()].m; a++) {
        A.cnts[a] = r_out[*labels.begin()].cnts[a];
        A.vals[a] = v ? *labels.begin() : 0;
        v = !v;
    }

    // then merge the other labels
    std::unique_ptr<uint[]> cnts = std::make_unique<uint[]>(h * w + 1);
    std::unique_ptr<int[]> vals = std::make_unique<int[]>(h * w + 1);
    for (auto label = ++labels.begin(); label != labels.end(); label++) {
        RLE B = r_out[*label];
        if (B.cnts == 0)
            continue;

        uint cnt_a = A.cnts[0];
        uint cnt_b = B.cnts[0];
        int next_val_a = A.vals[0];
        int val_a = next_val_a;
        int val_b = *label;
        bool next_vb = false;
        bool vb = next_vb;
        uint nb_seq_a, nb_seq_b;
        nb_seq_a = nb_seq_b = 1;
        int m = 0;

        int cnt_tot = 1;  // check if we advanced at all
        while (cnt_tot > 0) {
            uint c = std::min(cnt_a, cnt_b);
            cnt_tot = 0;
            // advance A
            cnt_a -= c;
            if (!cnt_a && nb_seq_a < A.m) {
                cnt_a = A.cnts[nb_seq_a];  // next sequence for A
                next_val_a = A.vals[nb_seq_a];
                nb_seq_a++;
            }
            cnt_tot += cnt_a;
            // advance B
            cnt_b -= c;
            if (!cnt_b && nb_seq_b < B.m) {
                cnt_b = B.cnts[nb_seq_b++];  // next sequence for B
                next_vb = !next_vb;
            }
            cnt_tot += cnt_b;

            if (val_a && vb) {
                vals[m] = (!cnt_a) ? val_a : val_b;
            } else if (val_a) {
                vals[m] = val_a;
            } else if (vb) {
                vals[m] = val_b;
            } else {
                vals[m] = 0;
            }
            cnts[m] = c;
            m++;

            // since we switched sequence for A or B, apply the new value from now on
            val_a = next_val_a;
            vb = next_vb;

            if (cnt_a == 0) break;
        }
        // copy back the buffers to the destination encoding
        A.m = m;
        for (int i = 0; i < m; i++) A.cnts[i] = cnts[i];
        for (int i = 0; i < m; i++) A.vals[i] = vals[i];
    }

    // Decode final pixelwise masks encoded via RLE and polygons
    memset(pixelwise_labels.data(), 0, h * w * sizeof(int));
    int x = 0, y = 0;
    for (uint i = 0; i < A.m; i++) {
        for (uint j = 0; j < A.cnts[i]; j++) {
            pixelwise_labels[x + y * w] = A.vals[i];
            if (++y >= h) {
                y = 0;
                x++;
            }
        }
    }

    // Destroy RLEs
    rlesFree(&r_out, *labels.rbegin() + 1);
    for (auto rles : FromPoly)
        for (auto &rle : rles.second)
            rleFree(&rle);
}

void COCOMetaDataReader::read_all(const std::string &path)
{
    _coco_metadata_read_time.start(); // Debug timing
    std::string rle_str;
    std::vector<uint32_t> rle_uints;
    uint32_t max_width = 0, max_height = 0;
    RLE *R = new RLE();
    std::ifstream f;
    f.open (path, std::ifstream::in|std::ios::binary);
    if (f.fail()) THROW("ERROR: Given annotations file not present " + path);
    f.ignore( std::numeric_limits<std::streamsize>::max() );
    auto file_size = f.gcount();
    f.clear();   //  Since ignore will have set eof.
    if (file_size == 0)
    { // If file is empty return
        f.close();
        THROW("ERROR: Given annotations file not valid " + path);
    }
    std::unique_ptr<char, std::function<void(char *)>> buff(
        new char[file_size + 1],
        [](char *data)
        { delete[] data; });
    f.seekg(0, std::ios::beg);
    buff.get()[file_size] = '\0';
    f.read(buff.get(), file_size);
    f.close();

    LookaheadParser parser(buff.get());

    BoundingBoxCords bb_coords;
    Labels bb_labels;
    ImgSizes img_sizes;
    std::vector<int> polygon_count;
    bool rle_flag = false;
    int polygon_size = 0;
    std::vector<std::vector<int>> vertices_count;

    BoundingBoxCord box;
    ImgSize img_size;
    RAPIDJSON_ASSERT(parser.PeekType() == kObjectType);
    parser.EnterObject();
    while (const char *key = parser.NextObjectKey())
    {
        if (0 == std::strcmp(key, "images"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue())
            {
                string image_name;
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "width"))
                    {
                        img_size.w = parser.GetInt();
                        max_width = std::max((uint32_t)img_size.w, max_width);
                    }
                    else if (0 == std::strcmp(internal_key, "height"))
                    {
                        img_size.h = parser.GetInt();
                        max_height = std::max((uint32_t)img_size.h, max_height);
                    }
                    else if (0 == std::strcmp(internal_key, "file_name"))
                    {
                        image_name = parser.GetString();
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
                _map_img_sizes.insert(pair<std::string, ImgSize>(image_name, img_size));
                img_size = {};
            }
        }
        else if (0 == std::strcmp(key, "categories"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            int id = 1;//continuous_idx = 1;;
            while (parser.NextArrayValue())
            {
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "id"))
                    {
                        id = parser.GetInt();
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
                //_label_info.insert(std::make_pair(id, continuous_idx));
                //continuous_idx++;
            }
        }
        else if (0 == std::strcmp(key, "annotations"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue())
            {
                int id = 1, label = 0, iscrowd = 0;
                std::array<double, 4> bbox;
                std::vector<float> mask;
                std::vector<int> vertices_array;
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "image_id"))
                    {
                        id = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "category_id"))
                    {
                        label = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "iscrowd"))
                    {
                        iscrowd = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "bbox"))
                    {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray();
                        int i = 0;
                        while (parser.NextArrayValue())
                        {
                            bbox[i] = parser.GetDouble();
                            ++i;
                        }
                    }
                    else if ((_output->get_metadata_type() == MetaDataType::PolygonMask || _output->get_metadata_type() == MetaDataType::PixelwiseMask) && 0 == std::strcmp(internal_key, "segmentation"))
                    {
                        if (parser.PeekType() == kObjectType && _output->get_metadata_type() == MetaDataType::PixelwiseMask)
                        {
                            parser.EnterObject();
                            rle_str.clear();
                            rle_uints.clear();
                            int h = -1, w = -1;
                            while (const char* another_key = parser.NextObjectKey()) {
                                if (0 == std::strcmp(another_key, "size")) {
                                    RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                                    parser.EnterArray();
                                    parser.NextArrayValue();
                                    h = parser.GetInt();
                                    parser.NextArrayValue();
                                    w = parser.GetInt();
                                    parser.NextArrayValue();
                                } else if (0 == std::strcmp(another_key, "counts")) {
                                    if (parser.PeekType() == kStringType) {
                                        rle_str = parser.GetString();
                                    } else if (parser.PeekType() == kArrayType) {
                                        parser.EnterArray();
                                        while (parser.NextArrayValue()) {
                                            rle_uints.push_back(parser.GetInt());
                                        }
                                    } else {
                                        parser.SkipValue();
                                    }
                                } else {
                                    parser.SkipValue();
                                }
                            }
                            if (!rle_str.empty()) {
                                rleInit(R, h, w, rle_uints.size(), const_cast<uint*>(rle_uints.data()));
                            } else if (!rle_uints.empty()) {
                                rleFrString(R, const_cast<char*>(rle_str.c_str()), h, w);
                            }
                            rle_flag = true;
                        }
                        else
                        {
                            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                            parser.EnterArray();
                            while (parser.NextArrayValue())
                            {
                                polygon_size += 1;
                                int vertex_count = 0;
                                parser.EnterArray();
                                while (parser.NextArrayValue())
                                {
                                    
                                    mask.push_back(parser.GetDouble());
                                    vertex_count += 1;
                                }
                                vertices_array.push_back(vertex_count);
                            }
                        }
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
                char buffer[13];
                sprintf(buffer, "%012d", id);
                string str(buffer);
                std::string file_name = str + ".jpg";

                auto it = _map_img_sizes.find(file_name);
                ImgSize image_size = it->second; //Normalizing the co-ordinates & convert to "ltrb" format
                if ((_output->get_metadata_type() == MetaDataType::PolygonMask || _output->get_metadata_type() == MetaDataType::PixelwiseMask) && iscrowd == 0)
                {
                    box.l = bbox[0];
                    box.t = bbox[1];
                    box.r = (bbox[0] + bbox[2] - 1);
                    box.b = (bbox[1] + bbox[3] - 1);
                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    polygon_count.push_back(polygon_size);
                    vertices_count.push_back(vertices_array);
                    add(file_name, bb_coords, bb_labels, image_size, mask, polygon_count, vertices_count);
                    mask.clear();
                    polygon_size = 0;
                    polygon_count.clear();
                    vertices_count.clear();
                    vertices_array.clear();
                    bb_coords.clear();
                    bb_labels.clear();
                }
                else if (!(_output->get_metadata_type() == MetaDataType::PolygonMask || _output->get_metadata_type() == MetaDataType::PixelwiseMask))
                {
                    box.l = bbox[0];
                    box.t = bbox[1];
                    box.r = (bbox[0] + bbox[2]);
                    box.b = (bbox[1] + bbox[3]);
                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    add(file_name, bb_coords, bb_labels, image_size, id);
                    bb_coords.clear();
                    bb_labels.clear();
                }
                if (rle_flag && _output->get_metadata_type() == MetaDataType::PixelwiseMask) {
                    generate_pixelwise_mask(file_name, R);
                    rleFree(R);
                    rle_flag = false;
                }
                image_size = {};
            }
        }
        else
        {
            parser.SkipValue();
        }
    }
    for (auto &elem : _map_content)
    {
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_labels();
        Labels continuous_label_id;
        for (unsigned int i = 0; i < bb_coords.size(); i++)
        {
            auto _it_label = _label_info.find(bb_labels[i]);
            int cnt_idx = _it_label->second;
            continuous_label_id.push_back(cnt_idx);
        }
        elem.second->set_labels(continuous_label_id);
        if (_output->get_metadata_type() == MetaDataType::PixelwiseMask) {
            std::vector<int>& pixelwise_label = elem.second->get_pixelwise_label();
            if (pixelwise_label.size() == 0) {
                //std::cout << "Gen1" << std::endl;
                generate_pixelwise_mask(elem.first, NULL);
            }
        }
    }
    delete(R);
    _max_width = max_width;
    _max_height = max_height;
    _coco_metadata_read_time.end(); // Debug timing
    //print_map_contents();
    // std::cout << "coco read time in sec: " << _coco_metadata_read_time.get_timing() / 1000 << std::endl;
}

void COCOMetaDataReader::release(std::string image_name)
{
    if (!exists(image_name))
    {
        WRN("ERROR: Given name not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void COCOMetaDataReader::release()
{
    _map_content.clear();
    _map_img_sizes.clear();
}

COCOMetaDataReader::COCOMetaDataReader() : _coco_metadata_read_time("coco meta read time", DBG_TIMING)
{
}
