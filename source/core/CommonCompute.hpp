//
//  CommonCompute.hpp
//  MNN
//
//  Created by MNN on 2021/07/23.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//


#ifndef CommonCompute_hpp
#define CommonCompute_hpp
#include <random>

namespace MNN {
/** 通用计算工具类 */
class MNN_PUBLIC CommonCompute {
public:
    // 稀疏相关的通用函数
    // sparse common functions
    
    /**
     * @brief 统计权重稀疏性
     * @tparam ElementType 元素类型
     * @param weightNNZElement 非零元素数量（输出）
     * @param weightBlockNumber 非零块数量（输出）
     * @param data 权重数据
     * @param h 输出通道数
     * @param l 每个输出通道的权重长度
     * @param sparseBlockOC 稀疏块大小（按输出通道数）
     * 
     * 统计权重的非零元素数量和非零块数量，支持按块稀疏
     */
    template <typename ElementType>
    static void statisticWeightSparsity(size_t& weightNNZElement, size_t& weightBlockNumber, const ElementType* data, size_t h, size_t l,  int sparseBlockOC) {

        size_t nnzBlock = 0;     // 非零块数量
        size_t nnzTail = 0;      // 剩余部分的非零元素数量
        int i = 0;
        
        // 处理完整的稀疏块
        for (; i + sparseBlockOC <= h; i += sparseBlockOC) {
            for(int j = 0; j < l; j += 1) {
                // 检查块是否全为零，非零则计数
                nnzBlock += !checkAllZeros(data, l, sparseBlockOC, 1);
                data++;
            }
            // 跳过当前块的其他元素
            data += l * (sparseBlockOC - 1);
        }
        
        // 处理剩余部分（不足一个完整块）
        for (; i < h; i++) {
            for(int j = 0; j < l; j++) {
                // 统计单个非零元素
                nnzTail += (*data != 0);
                data++;
            }
        }
        
        // 计算总非零元素数量和非零块数量
        weightNNZElement = nnzBlock * sparseBlockOC + nnzTail;
        weightBlockNumber = nnzBlock + nnzTail;
        return;
    }

    /**
     * @brief 生成指定稀疏度的随机权重
     * @tparam ElementType 元素类型
     * @param weightNNZElement 非零元素数量（输出）
     * @param weightBlockNumber 非零块数量（输出）
     * @param data 权重数据指针
     * @param oc 输出通道数
     * @param reduceDimLength 每个输出通道的权重长度
     * @param sparsity 稀疏度（0-1之间）
     * @param sparseBlockOC 稀疏块大小（按输出通道数）
     * @param minValue 随机值最小值
     * @param maxValue 随机值最大值
     * 
     * 生成指定稀疏度的随机权重，支持按块稀疏
     */
    template <typename ElementType>
    static void fillRandValueAsSparsity(size_t& weightNNZElement, size_t& weightBlockNumber, ElementType* data, int oc, int reduceDimLength, float sparsity, int sparseBlockOC, ElementType minValue = 0, ElementType maxValue = 1) {
        unsigned int seed = 1000;                  // 随机种子
        std::mt19937 rng(seed);                    // 随机数生成器
        std::uniform_real_distribution<float> uniform_dist(0, 1);     // 用于决定是否为零
        std::uniform_real_distribution<ElementType> uniform_value(minValue, maxValue); // 用于生成非零值
        
        size_t nnzBlock = 0;     // 非零块数量
        size_t nnzTail = 0;      // 剩余部分的非零元素数量
        int ocEven = (oc / sparseBlockOC) * sparseBlockOC;  // 完整块的输出通道数

        size_t ioc = 0;
        // 处理完整的稀疏块
        for (; ioc < ocEven; ioc += sparseBlockOC) {
            for (size_t i = 0; i < reduceDimLength; i++) {
                // 决定当前块是否为零
                bool isZero = uniform_dist(rng) <= sparsity;
                for (int iblock = 0; iblock < sparseBlockOC; iblock++) {
                    // 填充块内的值
                    *(data + iblock * reduceDimLength) = isZero ? 0.f : uniform_value(rng);
                }
                data++;
                // 统计非零块
                nnzBlock += !isZero;
            }
            // 跳过当前块的其他元素
            data += (sparseBlockOC - 1) * reduceDimLength;
        }
        
        // 处理剩余部分（不足一个完整块）
        for (; ioc < oc; ioc++) {
            for (size_t i = 0; i < reduceDimLength; i++) {
                // 决定当前元素是否为零
                bool isZero = uniform_dist(rng) <= sparsity;
                *data++ = isZero ? 0.f : uniform_value(rng);
                // 统计非零元素
                nnzTail += !isZero;
            }
        }
        
        // 计算总非零元素数量和非零块数量
        weightNNZElement = nnzBlock * sparseBlockOC + nnzTail;
        weightBlockNumber = nnzBlock + nnzTail;
    }
    
    /**
     * @brief 检查指定块是否全为零
     * @tparam ElementType 元素类型
     * @param source 数据指针
     * @param rowDimLength 行维度长度
     * @param blockRow 块行数
     * @param blockCol 块列数
     * @return 是否全为零
     * 
     * 检查指定大小的块是否所有元素都为零
     */
    template <typename ElementType>
    bool static checkAllZeros(const ElementType * source, size_t rowDimLength, int blockRow, int blockCol) {
        for (int i = 0; i < blockRow; i++) {
            for (int j = 0; j < blockCol; j++) {
                // 如果发现非零元素，返回false
                if (*(source + i * rowDimLength + j) != 0) {
                    return false;
                }
            }
        }
        // 所有元素都为零
        return true;
    }
    
    /**
     * @brief 将浮点权重压缩为稀疏格式
     * @param op 操作符
     * @return 是否成功压缩
     * 
     * 将浮点权重压缩为稀疏格式，只存储非零元素及其索引
     */
    static bool compressFloatWeightToSparse(MNN::OpT* op) {
        auto opType = op->type;
        auto param = op->main.AsConvolution2D();
        
        // 检查是否有稀疏参数
        if (param->sparseParameter.get() == nullptr) {
            return false;
        }
        
        // 编码稀疏浮点权重
        // Encode for sparse float weight
        size_t weightSize = param->weight.size();

        // 检查权重大小是否超过uint32_t的最大值
        if (weightSize > std::numeric_limits<uint32_t>().max()) {
            MNN_ERROR("The weightSize exceed uint32_t, can't compress the sparse weight\n");
            return false;
        }
        
        // 创建量化参数对象存储稀疏数据
        param->quanParameter.reset(new IDSTQuanT);
        std::vector<uint32_t> indexes;      // 非零元素索引
        std::vector<float> newWeights;       // 非零元素值

        // 遍历权重，收集非零元素
        for (size_t i=0; i<weightSize; ++i) {
            if (param->weight[i] != 0.0f) {
                indexes.emplace_back(i);
                newWeights.emplace_back(param->weight[i]);
            }
        }
        
        // 如果没有非零元素，添加一个零元素以避免错误
        // If empty, Add Single weight to avoid error, runtime can't extract full sparse convolution
        if (indexes.empty()) {
            indexes.emplace_back(0);
            newWeights.emplace_back(0.0f);
        }
        
        // 清空原始权重，存储稀疏数据
        param->weight.clear();
        param->quanParameter->alpha = std::move(newWeights);     // 存储非零值
        param->quanParameter->weightSize = (uint32_t)weightSize; // 原始权重大小
        param->quanParameter->index = std::move(indexes);        // 存储非零元素索引
        
        return true;
    }
};
} // namespace MNN

#endif
