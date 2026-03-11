//
//  IDSTEncoder.hpp
//  MNN
//
//  Created by MNN on 2021/02/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef IDSTENCODER_HPP
#define IDSTENCODER_HPP

#include <map>
#include <sstream>
#include "MNN_generated.h"
#include <cmath>

using namespace MNN;

namespace IDSTEncoder {

/**
 * @brief 写入blob维度信息
 * @param out 输出流
 * @param dims 维度数组
 * @return 是否使用32位整数
 * 
 * 将维度信息写入输出流，根据维度值大小选择使用16位或32位整数
 */
static bool WriteBlobDim(std::ostream &out, std::vector<int> dims)
{
    char tmp[4];
    bool useInt32 = false;
    // 写入维度数量
    ((unsigned char *)tmp)[0] = (unsigned char)dims.size();
    out.write(tmp, 1);
    // 检查是否需要使用32位整数
    for (int i = 0; i < dims.size(); i++) {
        if (dims[i] > ((1<<16)-1)) {
            useInt32 = true;
            break;
        }
    }
    // 根据维度值大小选择写入格式
    if (useInt32) {
        // 使用32位整数写入维度
        for (int i = 0; i < dims.size(); i++) {
            unsigned int tmpShort = (unsigned int)dims[i];
            out.write((const char*)(&tmpShort), 4);
        }
    } else {
        // 使用16位整数写入维度
        for (int i = 0; i < dims.size(); i++) {
            unsigned short tmpShort = (unsigned short)dims[i];
            out.write((const char*)(&tmpShort), 2);
        }
    }
    return useInt32;
}

/**
 * @brief 填充缓冲区
 * @param buf      目标缓冲区
 * @param buf_len  缓冲区长度
 * @param arr      源数组
 * @param arr_len  源数组长度
 * @param iNeedBits 需要的位数
 * 
 * 将源数组的值按指定位数填充到目标缓冲区中
 * 支持跨字节边界填充，使用位操作实现高效压缩
 */
static void FillBuffer(char *buf, unsigned int buf_len, const char *arr, unsigned int arr_len, unsigned char iNeedBits)
{
    // 清零目标缓冲区
    memset(buf, 0, buf_len);
    char *tmp = buf;
    int iOffset = 0;
    unsigned char cMask = (1 << iNeedBits) - 1; // 创建掩码，提取指定位数
    
    // 遍历源数组，将每个值填充到缓冲区
    for (int i = 0; i < arr_len; i++)
    {
        char value = arr[i];
        // 计算位移量，处理跨字节边界的情况
        int uShift = 8 - iNeedBits - iOffset % 8;
        if (uShift < 0)
        {
            // 跨越字节边界，需要写入两个字节
            tmp[iOffset / 8] |= ((value & cMask) >> (0 - uShift));
            tmp[(iOffset / 8) + 1] |= ((value & cMask) << (8 + uShift));
        }
        else
        {
            // 在同一字节内写入
            tmp[iOffset / 8] |= ((value & cMask) << uShift);
        }
        iOffset += iNeedBits;
        // 如果到达字节边界，移动到下一个字节
        if (iOffset % 8 == 0)
        {
            tmp += iOffset / 8;
            iOffset = 0;
        }
    }
}

/**
 * @brief 获取权重值集合
 * @param setWeight            输出权重值集合
 * @param weightData           权重数据
 * @param alphaData            量化参数（alpha值）
 * @param area                 权重面积
 * @param channel              通道数
 * @param asymmetricQuantFlag  是否使用非对称量化
 * @param bits                 量化位数
 * 
 * 根据权重数据和量化参数，计算所有可能的量化值
 * 支持对称和非对称量化两种模式
 */
static void GetWeightSet(std::set<int> &setWeight, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, const int bits)
{
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    setWeight.clear();
#define LINEAR_WEIGHT_SET
#ifdef LINEAR_WEIGHT_SET
    // using linear weight map
    // 使用线性权重映射，插入所有可能的值
    for (int i = min_value; i <= max_value; i++) {
        setWeight.insert(i);
    }
    return;
#endif
    // 非对称量化模式
    if (asymmetricQuantFlag) {
        for (int i = 0; i < channel; i++)
        {
            float min = alphaData[2*i];      // 最小值
            float alpha = alphaData[2*i+1]; // 缩放因子
            // 如果alpha太小，直接插入最小值
            if (alpha <= 1e-6f)
            {
                setWeight.insert(min_value);
                continue;
            }
            // 遍历该通道的所有权重，计算量化值
            for (int j = 0; j < area; j++)
            {
                float weight = weightData[i * area + j];
                // 量化公式：round((weight - min) / alpha) + min_value
                setWeight.insert(fmax(fmin(round((weight - min) / alpha) + min_value, max_value), min_value));
            }
        }
    } else {
        // 对称量化模式
        for (int i = 0; i < channel; i++)
        {
            float alpha = alphaData[i]; // 缩放因子
            // 如果alpha太小，直接插入0
            if (alpha <= 1e-6f)
            {
                setWeight.insert(0);
                continue;
            }
            // 遍历该通道的所有权重，计算量化值
            for (int j = 0; j < area; j++)
            {
                float weight = weightData[i * area + j];
                // 量化公式：round(weight / alpha)
                setWeight.insert(fmax(fmin(round(weight / alpha), max_value), min_value));
            }
        }
    }
}

/**
 * @brief 计算权重的稀疏度
 * @param weightData           权重数据
 * @param weightSize           权重大小
 * @param nnz                  非零元素数量（输出参数）
 * @param alphaData            量化参数
 * @param area                 权重面积
 * @param channel              通道数
 * @param asymmetricQuantFlag  是否非对称量化
 * @param bits                 量化位数
 * @param iMaxStep             最大步长（-1表示不限制）
 * @return 稀疏度（0-1之间）
 * 
 * 遍历所有权重，统计非零元素的数量，计算稀疏度
 * 支持对称和非对称量化，支持最大步长限制
 */
static float GetSparsity(const float* weightData, int weightSize, unsigned int& nnz, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, const int bits, int iMaxStep = -1)
{
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    nnz = 0;
    int iPreIdx = 0;
    float sparsity;
    // 非对称量化模式
    if (asymmetricQuantFlag) {
        for (int i = 0; i < weightSize; i++)
        {
            float min = alphaData[2*(i/area)];
            float alpha = alphaData[2*(i/area)+1];
            // 计算零点的量化值
            int zeroQuant = min_value;
            if (alpha > 1e-6) {
                zeroQuant = round((0.0f - min) / alpha) + min_value;
            }

            float weight = weightData[i];
            // 计算当前权重的量化值
            int value = min_value;
            if (alpha > 1e-6)
            {
                value = round((weight - min) / alpha) + min_value;
            }

            // 如果当前值不为零点，增加非零计数
            if (value != zeroQuant)
            {
                nnz++;
                iPreIdx = i;
            }
            // 如果距离上一个非零元素超过最大步长，增加非零计数（用于RLE压缩）
            if ((i - iPreIdx >= iMaxStep) && (iMaxStep != -1))
            {
                nnz++;
                iPreIdx = i;
            }
        }
    } else {
        // 对称量化模式
        for (int i = 0; i < weightSize; i++)
        {
            float alpha = alphaData[i / area];
            float weight = weightData[i];
            // 计算当前权重的量化值
            int value = 0;
            if (alpha > 1e-6f)
            {
                value = round(weight / alpha);
            }

            // 如果当前值不为0，增加非零计数
            if (value != 0)
            {
                nnz++;
                iPreIdx = i;
            }
            // 如果距离上一个非零元素超过最大步长，增加非零计数（用于RLE压缩）
            if ((i - iPreIdx >= iMaxStep) && (iMaxStep != -1))
            {
                nnz++;
                iPreIdx = i;
            }
        }
    }
    // 计算稀疏度：1 - (非零元素数 / 总元素数)
    sparsity = 1 - 1.0f * nnz / weightSize;
    return sparsity;
}

/**
 * @brief 获取最佳最大步长
 * @param weightData           权重数据
 * @param weightSize           权重大小
 * @param iMaxStepBits         最佳步长位数（输出参数）
 * @param BlobDataSize         blob数据大小
 * @param alphaData            量化参数
 * @param area                 权重面积
 * @param channel              通道数
 * @param asymmetricQuantFlag  是否非对称量化
 * @return 最佳非零元素数量
 * 
 * 遍历不同的步长位数（2-8位），计算每种情况下的压缩后大小
 * 选择压缩后大小最小的步长作为最佳步长
 */
static unsigned int GetBestMaxStep(const float* weightData, int weightSize, unsigned char& iMaxStepBits, int BlobDataSize, const float* alphaData, int area, int channel, bool asymmetricQuantFlag)
{
    size_t szBestSize = 1000000000;
    unsigned int best_nnz = 0;
    // 遍历不同的步长位数（2-8位）
    for (int i = 2; i < 9; i++)
    {
        unsigned int nnz = 0;
        // 计算在该步长下的非零元素数量
        GetSparsity(weightData, weightSize, nnz, alphaData, area, channel, asymmetricQuantFlag, BlobDataSize, pow(2, i) - 1);
        // 计算压缩后大小：步长索引大小 + 数据大小
        size_t tmp = ceil(0.125 * nnz * i) + ceil(0.125 * nnz * BlobDataSize);
        // 如果当前步长的压缩后大小更小，则更新最佳步长
        if (tmp < szBestSize)
        {
            iMaxStepBits = (unsigned char) i;
            szBestSize = tmp;
            best_nnz = nnz;
        }
    }
    return best_nnz;
}

/**
 * @brief 写入压缩的int8权重blob
 * @param out            输出流
 * @param weightData     int8权重数据
 * @param area           权重面积
 * @param channel        通道数
 * @param shapeUseInt32  是否使用32位整数存储形状（输出参数）
 * @param bits           量化位数
 * 
 * 将int8权重数据压缩后写入输出流
 * 压缩格式：1.形状 2.可用值数量 3.值集合 4.权重索引
 */
static void WriteCQBlobsInt8(std::ostream &out, const int8_t* weightData, int area, int channel, bool& shapeUseInt32, const int bits)
{
    //push values into buffer
    //Find int values in all blobs and check;
    std::set<int> setWeight;
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    setWeight.clear();
    // using linear weight map
    // 使用线性权重映射，插入所有可能的值
    for (int i = min_value; i <= max_value; i++) {
        setWeight.insert(i);
    }
    // 计算需要的位数
    int iCount = setWeight.size();
    int iNeedBits = ceil(log2(iCount));
    iNeedBits = iNeedBits < 1 ? 1 : iNeedBits;
    // 检查位数是否超过8
    if (iNeedBits > 8) {
        MNN_ERROR("The Bits need large than 8, the model may be error for user\n");
        return;
    }
    // 计算缓冲区长度
    size_t buf_len = size_t(ceil(0.125 * iNeedBits * area * channel));
    char *buf = new char[buf_len];
    {
        // 准备数据数组
        char *arr = new char[area * channel];
        unsigned char *tmp = (unsigned char*)arr;
        // 遍历所有权重，将值转换为正数索引
        for (int i = 0; i < channel; i++)
        {
            for (int j = 0; j < area; j++)
            {
                int value = weightData[i * area + j];
                *tmp = value + offset;
                tmp++;
            }
        }
        // 填充缓冲区
        FillBuffer(buf, buf_len, arr, area * channel, iNeedBits);
        delete[] arr;
    }
    //begin write to file
    {
        char tmp[100];
        //1. weights blob shape(unsigned int32)
        // 写入权重blob形状
        shapeUseInt32 = WriteBlobDim(out, {channel, area});
        // 2. Avalable values Count(unsigned char)
        // 写入可用值数量
        tmp[0] = (unsigned char)iCount;
        out.write(tmp, 1);
        // 3. valueset(signed char * valueset_size)
        // 写入值集合
        for (auto it = setWeight.begin(); it != setWeight.end(); it++)
        {
            tmp[0] = (unsigned char)*it;
            out.write(tmp, 1);
        }
        // 4. weights indexes(size = ceil(0.125*weights_count*ceil(log2(Avalable_values_Count))))
        // 写入权重索引
        out.write(buf, buf_len);
        //g_totalSize += 1 + setWeight.size() + buf_len;
    }
    delete[] buf;
}


/**
 * @brief 写入压缩的权重blob
 * @param out                  输出流
 * @param weightData           float权重数据
 * @param alphaData            量化参数
 * @param area                 权重面积
 * @param channel              通道数
 * @param asymmetricQuantFlag  是否非对称量化
 * @param shapeUseInt32        是否使用32位整数存储形状（输出参数）
 * @param bits                 量化位数
 * 
 * 将float权重数据量化并压缩后写入输出流
 * 支持对称和非对称量化两种模式
 * 压缩格式：1.形状 2.可用值数量 3.值集合 4.权重索引
 */
static void WriteCQBlobs(std::ostream &out, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, bool& shapeUseInt32, const int bits)
{
    //push values into buffer
    //Find int values in all blobs and check;
    // 获取权重值集合
    std::set<int> setWeight;
    GetWeightSet(setWeight, weightData, alphaData, area, channel, asymmetricQuantFlag, bits);
    // 计算需要的位数
    int iCount = setWeight.size();
    int iNeedBits = ceil(log2(iCount));
    iNeedBits = iNeedBits < 1 ? 1 : iNeedBits;
    // 检查位数是否超过8
    if (iNeedBits > 8) {
        MNN_ERROR("The Bits need large than 8, the model may be error for user\n");
        return;
    }
    // 创建权重值到索引的映射
    std::map<int, unsigned char> mapWeight;
    int iIdx = 0;
    for (std::set<int>::iterator it = setWeight.begin(); it != setWeight.end(); it++)
    {
        mapWeight[*it] = iIdx++;
    }
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    // 计算缓冲区长度
    size_t buf_len = size_t(ceil(0.125 * iNeedBits * area * channel));
    char *buf = new char[buf_len];
    {
        // 准备数据数组
        char *arr = new char[area * channel];
        unsigned char *tmp = (unsigned char*)arr;
        // 非对称量化模式
        if (asymmetricQuantFlag) {
            for (int i = 0; i < channel; i++)
            {
                float min = alphaData[2*i];
                float alpha = alphaData[2*i+1];
                for (int j = 0; j < area; j++)
                {
                    float weight = weightData[i * area + j];
                    // 计算量化值
                    int value = min_value;
                    if (alpha > 1e-6f)
                    {
                        value = fmax(fmin(round((weight - min) / alpha) + min_value, max_value), min_value);
                    }
#ifdef LINEAR_WEIGHT_SET
                    *tmp = value + offset;
#else
                    *tmp = mapWeight[value];
#endif
                    tmp++;
                }
            }
        } else {
            // 对称量化模式
            for (int i = 0; i < channel; i++)
            {
                float alpha = alphaData[i];
                for (int j = 0; j < area; j++)
                {
                    float weight = weightData[i * area + j];
                    // 计算量化值
                    int value = 0;
                    if (alpha > 1e-6f)
                    {
                        value = fmax(fmin(round(weight / alpha), max_value), min_value);
                    }
#ifdef LINEAR_WEIGHT_SET
                    *tmp = value + offset;
#else
                    *tmp = mapWeight[value];
#endif
                    tmp++;
                }
            }
        }
        // 填充缓冲区
        FillBuffer(buf, buf_len, arr, area * channel, iNeedBits);
        delete[] arr;
    }
    //begin write to file
    {
        char tmp[100];
        //1. weights blob shape(unsigned int32)
        // 写入权重blob形状
        shapeUseInt32 = WriteBlobDim(out, {channel, area});
        // 2. Avalable values Count(unsigned char)
        // 写入可用值数量
        tmp[0] = (unsigned char)iCount;
        out.write(tmp, 1);
        // 3. valueset(signed char * valueset_size)
        // 写入值集合
        for (auto it = setWeight.begin(); it != setWeight.end(); it++)
        {
            tmp[0] = (unsigned char)*it;
            out.write(tmp, 1);
        }
        // 4. weights indexes(size = ceil(0.125*weights_count*ceil(log2(Avalable_values_Count))))
        // 写入权重索引
        out.write(buf, buf_len);
        //g_totalSize += 1 + setWeight.size() + buf_len;
    }
    delete[] buf;
}

/**
 * @brief 写入稀疏量化的权重blob
 * @param out                  输出流
 * @param weightData           float权重数据
 * @param alphaData            量化参数
 * @param area                 权重面积
 * @param channel              通道数
 * @param asymmetricQuantFlag  是否非对称量化
 * @param shapeUseInt32        是否使用32位整数存储形状（输出参数）
 * @param bits                 量化位数
 * @return 是否成功写入
 * 
 * 将权重数据量化并使用稀疏格式压缩后写入输出流
 * 使用RLE（Run-Length Encoding）压缩稀疏权重
 * 压缩格式：1.形状 2.非零元素数 3.步长位数 4.步长索引 5.可用值数量 6.值集合 7.非零权重索引
 */
static bool WriteSparseQuanBlobs(std::ostream &out, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, bool& shapeUseInt32, const int bits)
{
    // 获取权重值集合
    std::set<int> setWeight;
    GetWeightSet(setWeight, weightData, alphaData, area, channel, asymmetricQuantFlag, bits);
    // 计算数据需要的位数
    int iDataNeedBits = ceil(log2(setWeight.size()));
    iDataNeedBits = iDataNeedBits < 1 ? 1 : iDataNeedBits;
    // 创建权重值到索引的映射
    std::map<int, unsigned char> mapWeight;
    {
        int iIdx = 0;
        for (auto it = setWeight.begin(); it != setWeight.end(); it++)
        {
            mapWeight[*it] = iIdx++;
        }
    }
    // 获取最佳最大步长和非零元素数
    unsigned int nnz = 0;
    int weightSize = area * channel;
    unsigned char iNeedBits;
    nnz = GetBestMaxStep(weightData, weightSize, iNeedBits, iDataNeedBits, alphaData, area, channel, asymmetricQuantFlag);
    if (nnz <= 0) {
        return false;
    }
    //weight buf
    // 计算数据缓冲区长度
    size_t data_buf_len = size_t(ceil(0.125 * iDataNeedBits * nnz));
    char* data_buf = new char[data_buf_len];
    //sparse COO buf
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    // 计算步长缓冲区长度
    size_t buf_len = size_t(ceil(0.125 * iNeedBits * nnz));
    char* buf = new char[buf_len];
    { //fill buf with step values;
        // 准备索引数组和数据数组
        unsigned char* arr_idx = new unsigned char[nnz];
        unsigned char* data_arr = new unsigned char[nnz];
        unsigned char* tmp = arr_idx;
        int iMaxStep = pow(2, iNeedBits) - 1;
        int iPreIdx = 0;
        unsigned char* dTmp = data_arr;
        // 非对称量化模式
        if (asymmetricQuantFlag) {
            for (int i = 0; i < weightSize; i++)
            {
                float min = alphaData[2*(i/area)];
                float alpha = alphaData[2*(i/area)+1];
                // 计算零点的量化值
                int zeroQuant = min_value;
                if (alpha > 1e-6) {
                    zeroQuant = round((0.0f - min) / alpha) + min_value;
                }

                float weight = weightData[i];
                // 计算当前权重的量化值
                int value = min_value;
                if (alpha > 1e-6)
                {
                    value = round((weight - min) / alpha) + min_value;
                }

                // 如果当前值不为零点，记录非零元素
                if (value != zeroQuant)
                {
                    *dTmp = mapWeight[value];
                    *tmp = i - iPreIdx;
                    iPreIdx = i;
                    tmp++;
                    dTmp++;
                }
                // 如果距离上一个非零元素超过最大步长，插入零点
                if (i - iPreIdx >= iMaxStep)
                {
                    *dTmp = mapWeight[zeroQuant];
                    *tmp = i - iPreIdx;
                    iPreIdx = i;
                    tmp++;
                    dTmp++;
                }
            }
        } else {
            // 对称量化模式
            for (int i = 0; i < weightSize; i++)
            {
                float alpha = alphaData[i / area];
                float weight = weightData[i];
                // 计算当前权重的量化值
                int value = 0;
                if (alpha > 1e-6f)
                {
                    value = round(weight / alpha);
                }

                // 如果当前值不为0，记录非零元素
                if (value != 0)
                {
                    *dTmp = mapWeight[value];
                    *tmp = i - iPreIdx;
                    iPreIdx = i;
                    tmp++;
                    dTmp++;
                }
                // 如果距离上一个非零元素超过最大步长，插入0
                if (i - iPreIdx >= iMaxStep)
                {
                    *dTmp = mapWeight[0];
                    *tmp = i - iPreIdx;
                    iPreIdx = i;
                    tmp++;
                    dTmp++;
                }
            }
        }
        // 填充步长缓冲区和数据缓冲区
        FillBuffer(buf, buf_len, (char*) arr_idx, nnz, iNeedBits);
        FillBuffer(data_buf, data_buf_len, (char*) data_arr, nnz, iDataNeedBits);
        delete[] arr_idx;
        delete[] data_arr;
    }
    { //write
        char tmp[100];
        // 1.weights blob shape(unsigned int32)
        // 写入权重blob形状
        shapeUseInt32 = WriteBlobDim(out, {channel, area});
        // 2. nnz
        // 写入非零元素数量
        out.write((const char*) &nnz, 4);
        // 3. max_step use # bits () (unsigned char)
        // 写入步长位数
        out.write((const char*) &iNeedBits, 1);
        // 4. buf for steps ceil(nnz*step need bits/8)
        // 写入步长索引
        out.write(buf, buf_len);
        // 5. Avalable values Count(unsigned char)
        // 写入可用值数量
        tmp[0] = (unsigned char) setWeight.size();
        out.write(tmp, 1);
        // 6. valueset(signed char * valueset_size)
        // 写入值集合
        for (auto it = setWeight.begin(); it != setWeight.end(); it++)
        {
            tmp[0] = (unsigned char) *it;
            out.write(tmp, 1);
        }
        // 7. none zero weights indexes(nnz*ceil(log2(Avalable_values_Count))/8)
        // 写入非零权重索引
        out.write((const char*) data_buf, data_buf_len);
    }
    delete[] buf;
    delete[] data_buf;
    return true;
}

/**
 * @brief 编码权重数据
 * @param weight               float权重数据
 * @param scale                量化参数（scale值）
 * @param kernelSize           卷积核大小
 * @param kernelNum            卷积核数量
 * @param asymmetricQuantFlag  是否非对称量化
 * @param quantWeightPtr       int8量化权重指针（可选）
 * @param clampMin             最小钳位值
 * @param bits                 量化位数（默认8）
 * @param detectSparse         是否检测稀疏性（默认true）
 * @return 编码后的IDSTQuanT对象
 * 
 * 将权重数据进行量化压缩编码，支持多种压缩格式：
 * - type=1: 普通压缩量化（CQ）
 * - type=2: 稀疏压缩量化（SQ）
 * 
 * 如果detectSparse为true，会比较CQ和SQ两种格式的压缩大小，选择更小的格式
 */
static std::unique_ptr<IDSTQuanT> encode(const float* weight, const std::vector<float>& scale, int kernelSize, int kernelNum,
                                         bool asymmetricQuantFlag, const int8_t* quantWeightPtr, const int clampMin, const int bits = 8, bool detectSparse = true) {
    // compute block_size
    // 计算块大小和块数量
    auto alpha_size = scale.size();
    auto block_size = kernelSize;
    auto block_num = 1;
    // 如果是非对称量化，alpha数组大小是通道数的两倍（每个通道有min和alpha两个参数）
    if (asymmetricQuantFlag) {
        alpha_size /= 2;
    }
    // 如果alpha数量大于通道数，说明使用了分块量化
    if (alpha_size > kernelNum) {
        block_num = alpha_size / kernelNum;
        block_size = kernelSize / block_num;
    }
    bool shapeUseInt32 = false;
    std::unique_ptr<IDSTQuanT> idst(new IDSTQuanT);
    std::ostringstream outputStringStreamCQ;
    idst->aMaxOrBits = bits;
    // 如果提供了int8量化权重且没有float权重，直接使用int8权重
    if (quantWeightPtr && nullptr == weight) {
        WriteCQBlobsInt8(outputStringStreamCQ, quantWeightPtr, kernelSize, kernelNum, shapeUseInt32, bits);
        auto cqStr = outputStringStreamCQ.str();
        idst->type = 1;
        idst->buffer.resize(cqStr.size());
        ::memcpy(idst->buffer.data(), cqStr.data(), cqStr.size());
    } else {
        // 使用float权重进行量化压缩
        WriteCQBlobs(outputStringStreamCQ, weight, scale.data(), kernelSize, kernelNum, asymmetricQuantFlag, shapeUseInt32, bits);
        auto cqStr = outputStringStreamCQ.str();
        // 如果需要检测稀疏性，比较CQ和SQ两种格式的压缩大小
        if (detectSparse) {
            std::ostringstream outputStringStreamSQ;
            bool sparseValid = WriteSparseQuanBlobs(outputStringStreamSQ, weight, scale.data(), kernelSize, kernelNum, asymmetricQuantFlag, shapeUseInt32, bits);
            auto sqStr = outputStringStreamSQ.str();
            int int8Size = kernelNum * kernelSize;
            // 选择压缩后大小更小的格式
            if (cqStr.size() <= sqStr.size() || (!sparseValid)) {
                idst->type = 1;
                idst->buffer.resize(cqStr.size());
                ::memcpy(idst->buffer.data(), cqStr.data(), cqStr.size());
            } else {
                idst->type = 2;
                idst->buffer.resize(sqStr.size());
                ::memcpy(idst->buffer.data(), sqStr.data(), sqStr.size());
            }
        } else {
            // 不检测稀疏性，直接使用CQ格式
            idst->type = 1;
            idst->buffer.resize(cqStr.size());
            ::memcpy(idst->buffer.data(), cqStr.data(), cqStr.size());
        }
    }
    // 设置形状是否使用32位整数
    idst->shapeInt32 = shapeUseInt32;
    // 复制量化参数
    idst->alpha.resize(scale.size());
    ::memcpy(idst->alpha.data(), scale.data(), scale.size() * sizeof(float));
    idst->quantScale = 1.f;
    // 如果是非对称量化，设置读取类型和最小值
    if (asymmetricQuantFlag) {
        idst->readType = kernelNum;
        idst->aMin = clampMin;
    }
    return idst;
}

} // namespace IDSTEncoder

#endif // IDSTENCODER_HPP
