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

#include <cassert>
#include <algorithm>
#include <commons.h>
#include "file_list_reader.h"
#include <boost/filesystem.hpp>
#include <fstream>
#include <cmath>

namespace filesys = boost::filesystem;

FileListReader::FileListReader():
_shuffle_time("shuffle_time", DBG_TIMING)
{
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _file_id = 0;
    _shuffle = false;
    _file_count_all_shards = 0;
}

unsigned FileListReader::count_items()
{
    if(_loop)
        return _file_names.size();
    int ret = ((int)_file_names.size() -_read_counter);
    return ((ret < 0) ? 0 : ret);
}

size_t
FileListReader::last_batch_padded_size()
{
    return _last_batch_padded_size;
}

Reader::Status FileListReader::initialize(ReaderConfig desc)
{
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _file_list_path = desc.json_path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _last_batch_info = desc.get_last_batch_policy();
    _stick_to_shard = desc.get_stick_to_shard();
    _shard_size = desc.get_shard_size();
    _meta_data_reader = desc.meta_data_reader();
    ret = subfolder_reading();
    //shuffle dataset if set
    _shuffle_time.start();
    if( ret==Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();
    return ret;

}

void FileListReader::incremenet_read_ptr()
{
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
    if(_last_batch_info.first == RocalBatchPolicy::DROP)
    {
        if((_file_names.size() / _batch_count) == _curr_file_idx) // Check if its last batch
        {
            _curr_file_idx += _batch_count;
            _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
        }
    }
}
size_t FileListReader::open()
{
    auto file_path = _file_names[_curr_file_idx];// Get next file name
    incremenet_read_ptr();
    _last_file_path = _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        _last_id.erase(0, last_slash_idx + 1);
    }

    _current_fPtr = fopen(file_path.c_str(), "rb");// Open the file,

    if(!_current_fPtr) // Check if it is ready for reading
        return 0;

    fseek(_current_fPtr, 0 , SEEK_END);// Take the file read pointer to the end

    _current_file_size = ftell(_current_fPtr);// Check how many bytes are there between and the current read pointer position (end of the file)

    if(_current_file_size == 0)
    { // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, 0 , SEEK_SET);// Take the file pointer back to the start

    return _current_file_size;
}

size_t FileListReader::read_data(unsigned char* buf, size_t read_size)
{
    if(!_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int FileListReader::close()
{
    return release();
}

FileListReader::~FileListReader()
{
    release();
}

int
FileListReader::release()
{
    if(!_current_fPtr)
        return 0;
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
    return 0;
}

void FileListReader::reset()
{
    _shuffle_time.start();
    if (_shuffle) std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();
    if (_stick_to_shard == true) { 
        if (_shard_size > 0)
        {
        // Reset the variables
            _last_batch_padded_size = _read_counter = _curr_file_idx = 0;
            int read_start_index = 0;
            if (_shard_size < _batch_count)
                read_start_index = _batch_count;
            else{
                if(_shard_size % _batch_count)
                    read_start_index = _shard_size;
                else
                    read_start_index = _shard_size + (_shard_size % _batch_count);
            }
            // To re-arrange the starting index of file-loading keeping in mind the shard_size and batch size
            std::rotate(_file_names.begin(), _file_names.begin() + read_start_index, _file_names.end());
        } else {
        _curr_file_idx = 0;
        _read_counter = 0;
        }
    }
    else if (_stick_to_shard == false && _shard_count > 1) {
            // Reset the variables 
            _last_batch_padded_size = _in_batch_read_count = _curr_file_idx = _file_id = _read_counter = 0;
            _file_names.clear();
            increment_shard_id();
            generate_file_names(); // generates the data from next shard in round-robin fashion after completion of an epoch
            if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count) {
                // This is to pad within a batch in a shard. Need to change this according to fill / drop or partial.
                // Adjust last batch only if the last batch padded is true.
                replicate_last_image_to_fill_last_shard();
                LOG("FileReader in reset function - Replicated " + _folder_path + _last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
            }
            if (!_file_names.empty())
                LOG("FileReader in reset function - Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    }
    else {
        _curr_file_idx = 0;
        _read_counter = 0;
    }
}

void FileListReader::increment_shard_id()
{
    _shard_id = (_shard_id + 1) % _shard_count;
}

void FileListReader::generate_file_names()
{
    std::ifstream fp (_file_list_path);
    if (fp.is_open())  {
        while (fp) {
            std::string file_label_path;
            std::getline (fp, file_label_path);
            std::istringstream ss(file_label_path);
            std::string file_path;
            std::getline(ss, file_path, ' ');
            file_path = _folder_path + "/"+ file_path;
            if (filesys::is_regular_file(file_path )) {
                if(get_file_shard_id() != _shard_id ) {
                    _file_count_all_shards++;
                    incremenet_file_id();
                    continue;
                }
                _in_batch_read_count++;
                _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
                _last_file_name = file_path;
                _file_names.push_back(file_path);
                _file_count_all_shards++;
                incremenet_file_id();
            }
        }
    }
        uint images_to_pad_shard = _file_count_all_shards - (ceil(_file_count_all_shards / _shard_count) * _shard_count);
        if(!images_to_pad_shard) {
            for(int i = 0; i < images_to_pad_shard; i++) {
                if(get_file_shard_id() != _shard_id )
                {
                    _file_count_all_shards++;
                    incremenet_file_id();
                    continue;
                }
                _last_file_name = _file_names.at(i);
                _file_names.push_back(_last_file_name);
                _file_count_all_shards++;
                incremenet_file_id();
            }

    }
}

Reader::Status FileListReader::subfolder_reading()
{
    std::vector<std::string> entry_name_list;
    auto ret = Reader::Status::OK;
    generate_file_names();
    if(_file_names.empty())
        WRN("FileReader ShardID ["+ TOSTR(_shard_id)+ "] Did not load any file from " + _folder_path)
    if(_in_batch_read_count > 0 && _in_batch_read_count < _batch_count) {
        // This is to pad within a batch in a shard. Need to change this according to fill / drop or partial.
        // Adjust last batch only if the last batch padded is true.
        replicate_last_image_to_fill_last_shard();
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Replicated " + _folder_path + _last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count) ) + " times to fill the last batch")
    }
    if(!_file_names.empty())
        LOG("FileReader ShardID ["+ TOSTR(_shard_id)+ "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path )
    return ret;
}
void FileListReader::replicate_last_image_to_fill_last_shard()
{
    if(_last_batch_info.first == RocalBatchPolicy::BATCH_FILL) {
        if(_last_batch_info.second == true) {
        for(size_t i = (_batch_count - _in_batch_read_count); i < _batch_count; i++)
            _file_names.push_back(_last_file_name);
        } else  {
        for(size_t i = 0; i < (_batch_count - _in_batch_read_count); i++)
            _file_names.push_back(_file_names.at(i));
        }
    }
    // // drop
    else if(_last_batch_info.first == RocalBatchPolicy::DROP)
    {
        for(size_t i = 0; i < _in_batch_read_count; i++)
            _file_names.pop_back();
    }
    else if(_last_batch_info.first == RocalBatchPolicy::PARTIAL)
    {
        _last_batch_padded_size = _batch_count - _in_batch_read_count;
        if(_last_batch_info.second == true) {
        for(size_t i = (_batch_count - _in_batch_read_count); i < _batch_count; i++)
            _file_names.push_back(_last_file_name);
        } else  {
        for(size_t i = 0; i < (_batch_count - _in_batch_read_count); i++)
            _file_names.push_back(_file_names.at(i));
        }
    }

}

void FileListReader::replicate_last_batch_to_pad_partial_shard()
{
    if (_file_names.size() >=  _batch_count) {
        for (size_t i = 0; i < _batch_count; i++)
            _file_names.push_back(_file_names[i - _batch_count]);
    }
}


Reader::Status FileListReader::open_folder()
{
    if ((_src_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);

    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;

        if(get_file_shard_id() != _shard_id )
        {
            _file_count_all_shards++;
            incremenet_file_id();
            continue;
        }
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count%_batch_count == 0) ? 0 : _in_batch_read_count;
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _last_file_name = file_path;
        _file_names.push_back(file_path);
        _file_count_all_shards++;
        incremenet_file_id();
    }
    uint images_to_pad_shard = _file_count_all_shards - (ceil(_file_count_all_shards / _shard_count) * _shard_count);
    if(!images_to_pad_shard) {
        for(int i = 0; i < images_to_pad_shard; i++) {
            if(get_file_shard_id() != _shard_id )
            {
                _file_count_all_shards++;
                incremenet_file_id();
                continue;
            }
            _last_file_name = _file_names.at(i);
            _file_names.push_back(_last_file_name);
            _file_count_all_shards++;
            incremenet_file_id();
        }
    }
    if(_file_names.empty())
        WRN("FileReader ShardID ["+ TOSTR(_shard_id)+ "] Did not load any file from " + _folder_path)

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t FileListReader::get_file_shard_id()
{
    if(_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    //return (_file_id / (_batch_count)) % _shard_count;
    return _file_id  % _shard_count;
}