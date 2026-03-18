
#ifndef MEMORY_FORMATER_H
#define MEMORY_FORMATER_H

#include "MNN/MNNDefine.h"
#include <vector>

/**
 * @brief 打印维度信息
 * @param dims 维度数组
 * 
 * 以格式化的方式打印维度信息，格式为 {dim1, dim2, ..., dimn}
 */
inline void printDims(const std::vector<int>& dims) {
  int num_dims = dims.size();
  MNN_PRINT(" {");
  if (num_dims > 0) MNN_PRINT("%d", dims.at(0));
  for (size_t i = 1; i < num_dims; i++) {
    MNN_PRINT(", %d", dims.at(i));
  }
  MNN_PRINT("}");
}

/**
 * @brief 将BF16格式转换为FP32格式
 * @param s16Value BF16格式的16位整数值
 * @return 转换后的FP32浮点数
 * 
 * BF16 (Brain Float 16) 是一种16位浮点数格式，
 * 这里通过位操作将其转换为32位浮点数
 */
inline float MNNBF16ToFP32(int16_t s16Value) {
    int32_t s32Value = ((int32_t)s16Value) << 16;
    float* fp32Value = (float*)(&s32Value);
    return *fp32Value;
}

/**
 * @brief 格式化打印float类型值
 * @param prefix 前缀字符串
 * @param value 要打印的float值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const float& value, const char* suffix) {
    MNN_PRINT("%s%f%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印double类型值
 * @param prefix 前缀字符串
 * @param value 要打印的double值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const double& value, const char* suffix) {
   MNN_PRINT("%s%f%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印uint8_t类型值
 * @param prefix 前缀字符串
 * @param value 要打印的uint8_t值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const uint8_t& value, const char* suffix) {
   MNN_PRINT("%s%d%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印int8_t类型值
 * @param prefix 前缀字符串
 * @param value 要打印的int8_t值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const int8_t& value, const char* suffix) {
   MNN_PRINT("%s%d%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印int16_t类型值（BF16格式）
 * @param prefix 前缀字符串
 * @param value 要打印的int16_t值（BF16格式）
 * @param suffix 后缀字符串
 * 
 * 将BF16格式的int16_t转换为float后打印
 */
inline void formatPrint(const char* prefix, const int16_t& value, const char* suffix) {
   MNN_PRINT("%s%f%s", prefix, MNNBF16ToFP32(value), suffix);
}

/**
 * @brief 格式化打印int类型值
 * @param prefix 前缀字符串
 * @param value 要打印的int值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const int& value, const char* suffix) {
   MNN_PRINT("%s%d%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印unsigned int类型值
 * @param prefix 前缀字符串
 * @param value 要打印的unsigned int值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const unsigned int& value, const char* suffix) {
   MNN_PRINT("%s%u%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印long int类型值
 * @param prefix 前缀字符串
 * @param value 要打印的long int值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const long int& value, const char* suffix) {
   MNN_PRINT("%s%ld%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印unsigned long类型值
 * @param prefix 前缀字符串
 * @param value 要打印的unsigned long值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const unsigned long& value, const char* suffix) {
   MNN_PRINT("%s%lu%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印long long类型值
 * @param prefix 前缀字符串
 * @param value 要打印的long long值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const long long& value, const char* suffix) {
   MNN_PRINT("%s%lld%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印unsigned long long类型值
 * @param prefix 前缀字符串
 * @param value 要打印的unsigned long long值
 * @param suffix 后缀字符串
 */
inline void formatPrint(const char* prefix, const unsigned long long& value, const char* suffix) {
   MNN_PRINT("%s%llu%s", prefix, value, suffix);
}

/**
 * @brief 格式化打印矩阵数据
 * @tparam ElementType 元素类型
 * @param data 数据指针
 * @param dims 维度数组
 * 
 * 递归打印多维矩阵数据，支持标量、向量、矩阵等不同维度的数据
 * 当数据量超过MaxLines时，会跳过中间部分，只打印前MaxLines和后MaxLines的数据
 */
template <typename ElementType>
inline void formatMatrix(ElementType* data, std::vector<int> dims) {

  const int MaxLines = 100; // 最大打印行数

  MNN_PRINT("shape:");
  printDims(dims);
  MNN_PRINT("\n");
  
  // 移除末尾维度为1的维度
  while (dims.size() > 1) {
    if (*(dims.end() - 1) == 1) {
        dims.erase(dims.end() - 1);
    } else {
        break;
    }
  }

  // 处理标量情况
  if (dims.size() == 0) {
    formatPrint("scalar:", *data, "\n");
    return;
  }
  
  int highDim = dims[0];
  const int lines = highDim < MaxLines ? highDim : MaxLines; // 前lines行
  const int tailStart = highDim - MaxLines > lines ? highDim - MaxLines : lines; // 从tailStart开始的后MaxLines行

  // 处理一维向量
  if (dims.size() == 1) { // output elements in the last dim in a row.
    MNN_PRINT("{");
    for (int i = 0; i < lines; i++) {
      formatPrint("", data[i], ", ");
    }
    if (tailStart > lines) {
      formatPrint(", …skip middle ", tailStart - lines, "…,");
    }
    for (int i = tailStart; i < highDim; i++) {
      formatPrint("", data[i], ", ");
    }
    MNN_PRINT("}");
    return;

  } else {
    // 处理多维矩阵
    dims.erase(dims.begin()); // 移除第一个维度，递归处理剩余维度

    // 计算每个元素的步长
    int step = dims[0];
    for (size_t i = 1; i < dims.size(); i++) {
      step *= dims[i];
    }

    // 打印前lines个元素
    for (int i = 0; i < lines; i++) {
      formatPrint("", i, " th:");
      formatMatrix(data + i * step, dims);
      MNN_PRINT("\n");
    }
    // 如果数据量太大，打印跳过信息
    if (tailStart > lines) {
      formatPrint("{…skip middle ", tailStart - lines, " …}\n");
    }

    // 打印后MaxLines个元素
    for (int i = tailStart; i < highDim; i++) {
      formatPrint("", i, " th:");
      formatMatrix(data + i * step, dims);
      MNN_PRINT("\n");
    }

    return;

  }
}

#endif


