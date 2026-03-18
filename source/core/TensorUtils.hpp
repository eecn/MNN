//
//  TensorUtils.hpp
//  MNN
//
//  Created by MNN on 2019/01/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#ifndef TensorUtils_hpp
#define TensorUtils_hpp

#include <MNN/Tensor.hpp>
#include "Backend.hpp"
#include "AutoStorage.h"
#include "Tensor_generated.h"
#define MNN_MAX_TENSOR_DIM 9

#ifdef CONSTANT
#undef CONSTANT
#endif // CONSTANT

namespace MNN {
/** 张量数组属性 */
struct TensorArrayAttr {
    // 数组大小是否动态
    bool isDynamicSize = false;
    // 元素形状是否相同
    bool isIdenticalShape = false;
    // 元素数量
    uint32_t arraySize = 0;
    // 元素形状
    std::vector<std::vector<int>> elemShape;
};

/** 量化属性 */
struct QuantAttr {
    float scale;           // 量化缩放因子
    float zero = 0.0f;     // 零点
    float min  = -127.0f;  // 最小值
    float max  = 127.0f;   // 最大值
    DataType type = DataType_DT_INT8;  // 数据类型
};

/** 张量内部描述 */
struct Tensor::InsideDescribe {
    /** 视图结构 */
    struct View {
        int32_t offset = 0;      // 偏移量
        int32_t stride[3] = {1, 1, 1};  // 步长
    };
    
    /** 区域结构 */
    struct Region {
        View src;         // 源视图
        View dst;         // 目标视图
        int32_t size[3] = {1, 1, 1};  // 大小
        Tensor* origin;   // 原始张量
    };
    
    /** 填充结构 */
    struct pad {
        int32_t left = 0;    // 左边填充
        int32_t right = 0;   // 右边填充
        int32_t bottom = 0;  // 底部填充
        int32_t top = 0;     // 顶部填充
    };
    
    /** 内存类型 */
    enum MemoryType {
        /** 张量内存来自后端 */
        MEMORY_BACKEND = 0,

        /** 主机内存是否由张量拥有 */
        MEMORY_HOST,

        /** 张量没有内存 */
        MEMORY_VIRTUAL,

        /** 外部主机内存 */
        MEMORY_OUTSIDE,
    };
    
    /** 使用类型 */
    enum Usage {
        NORMAL,         // 普通
        INPUT,          // 输入
        OUTPUT,         // 输出
        CONSTANT,       // 常量
        /** 张量是否为可训练参数。可训练参数应存储在不同区域。 */
        TRAINABLE,      // 可训练
    };
    
    // 阶段信息掩码
    enum StageInfo {
        GEOMETRY_STAGE = 1,             // 几何阶段
        CONVERTED_STAGE = 1 << 1,       // 转换阶段
        COMPUTE_SHAPE_STAGE = 1 << 2,   // 计算形状阶段
        CONTENT_NOT_CHANGE = 1 << 3,    // 内容未更改
    };
    
    /** 额外张量信息容器 */
    struct NativeInsideDescribe {
    public:
        /** 维度格式 */
        MNN_DATA_FORMAT dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        
        union {
            /** 分离内存偏移量 */
            int offset;

            /** 用于释放句柄的函数 */
            void (*handleFreeFunction)(void*);
        } extra;
        
        MemoryType memoryType = MEMORY_BACKEND;  // 内存类型
        std::weak_ptr<Command> rasterCommand;     // 光栅命令
        /** 仅用于设备张量 */
        int useCount = 0;                         // 使用计数
        Usage usage = NORMAL;                     // 使用类型
        std::vector<Region> regions;              // 区域列表
        halide_dimension_t dims[MNN_MAX_TENSOR_DIM];  // 维度信息
        
        // 张量数组属性
        std::shared_ptr<TensorArrayAttr> tensorArrayAttr;
        // 张量量化属性
        std::shared_ptr<QuantAttr> quantAttr;
        
        bool applyQuant = false;    // 是否应用量化
        bool isMutable = true;      // 是否可变
        bool overlap = false;       // 仅用于strideSliceWrite
        int index = -1;             // 索引
        int group = 0;              // 分组
	int channel_pack_num = 4;    // 通道打包数量
        bool support_pack16 = true; // 是否支持16位打包
        pad mPads;                  // 填充信息
        
        // 对于isMutable = false的张量，确定内容是否可以转换到主后端
        uint32_t stageMask = 0;     // 阶段掩码
        
        // 用于共享内存
        SharedPtr<Backend::MemObj> mSharedMem;  // 共享内存对象
    };
    
    std::shared_ptr<NativeInsideDescribe> mContent;  // 内容描述
    SharedPtr<Backend::MemObj> mem;                  // 内存对象
    
    /** 获取后端 */
    inline Backend* getBackend() const {
        return backend;
    }
    
    /** 设置后端 */
    inline void setBackend(Backend* bn) {
        backend = bn;
    }
    
private:
    /** 仅用于设备张量。用于管理张量设备内存的后端。 */
    Backend* backend = nullptr;
};

typedef Tensor::InsideDescribe::Usage TensorUsage;

/** 张量工具类 */
class MNN_PUBLIC TensorUtils {
public:
    /**
     * @brief 获取张量的额外信息
     * @param tensor    给定的张量
     * @return 张量的额外信息
     */
    static Tensor::InsideDescribe::NativeInsideDescribe* getDescribe(const Tensor* tensor);

    /** 获取张量的原始描述 */
    static Tensor::InsideDescribe* getDescribeOrigin(const Tensor* tensor);

    /**
     * @brief 从源张量复制形状到目标张量
     * @param source        形状提供张量
     * @param dest          形状接收张量
     * @param copyFormat    是否复制数据格式
     * @param copyRef       是否复制引用
     */
    static void copyShape(const Tensor* source, Tensor* dest, bool copyFormat = false, bool copyRef = false);

    /**
     * @brief 从普通整数向量设置目标张量的形状
     * @param dest          形状接收张量
     * @param alldims       维度信息
     */
    static void setShape(Tensor* dest, const std::vector<int>& alldims);

    /**
     * 根据大小和重排序标志自动更新张量的步长
     * @param tensor    给定的张量
     */
    static void setLinearLayout(Tensor* tensor);

    /**
     * @brief 比较张量与预期值，允许一定误差
     * @param compareTensor 比较张量
     * @param toTensor      预期张量
     * @param tolerance     可容忍的误差，小于此值的误差将被忽略
     *                      对于整数类型，使用 `abs(v1 - v2) > tolerance` 比较
     *                      对于浮点类型，见 `overallTolerance`
     * @param overall       仅对浮点类型有效。如果为true，使用 `abs(v1 - v2) / max(abs(allExpectValues))` 比较
     *                      否则使用 `abs(v1 - v2) / abs(v2)`
     * @param printsError   是否打印错误数据
     * @param printsTensors 遇到错误时是否打印张量数据
     * @return 是否在误差范围内相等
     */
    static bool compareTensors(const Tensor* compareTensor, const Tensor* toTensor, float tolerance = 0,
                               bool overall = false, bool printsError = true, bool printsTensors = false);

    /** 设置张量信息 */
    static void setupTensorInfo(const Tensor* tensor, Tensor* wrapTensor, MNN_DATA_FORMAT mMidFormat);
    
    /** 创建完整切片 */
    static Tensor::InsideDescribe::Region makeFullSlice(Tensor* input);
    
    /** 创建完整引用 */
    static void makeFullRef(Tensor* output, Tensor* input);
    
    /** 检查区域是否完整 */
    static bool regionIsFull(Tensor* input);
    
    /** 检查区域是否为复制区域 */
    static bool isCopyRegion(const Tensor::InsideDescribe::Region& region);
    
    /** 检查区域是否为转置区域 */
    static bool isTransposeRegion(const Tensor::InsideDescribe::Region& region);
    
    /** 检查区域是否为平铺区域 */
    static bool isTileRegion(const Tensor::InsideDescribe::Region& region);
    
    /** 检查是否为深度到空间区域 */
    static bool isDepthToSpaceRegions(const Tensor* output);
    
    /** 重塑切片 */
    static bool reshapeSlice(Tensor::InsideDescribe::Region& slice, int outside, int inside, int axis);

    class FuseRegionStatus;
    
    /** 融合包装类 */
    class MNN_PUBLIC FuseWrap {
    public:
        FuseWrap();
        ~ FuseWrap();
        /** 匹配源区域和目标区域 */
        bool match(const Tensor::InsideDescribe::Region& srcReg, const Tensor::InsideDescribe::Region& dstReg);
        /** 应用区域融合 */
        void apply(const Tensor::InsideDescribe::Region& srcReg, Tensor::InsideDescribe::Region& dstReg);
    private:
        FuseRegionStatus* mStatus;
    };
    
    /** 调整张量以提高兼容性 */
    static void adjustTensorForCompability(Tensor* t);
    
    /** 获取张量的维度类型 */
    static Tensor::DimensionType getDimType(const Tensor* t);
    
    /** 获取张量的量化信息 */
    static std::vector<float> getQuantInfo(const Tensor* t);

    /** 获取张量的原始大小 */
    static size_t getRawSize(const Tensor* t);
    
    /** 设置光栅输入 */
    static void setRasterInputs(Command* cmd);

    /** 引用张量内容 */
    static bool refTensorContent(Tensor* dst, const Tensor* src);

    /** 获取张量的通道打包数 */
    static int getTensorChannelPack(const Tensor* tensor);

    /** 设置张量的通道打包数 */
    static void setTensorChannelPack(const Tensor* tensor, int pack);

    /** 设置张量是否支持打包 */
    static void setTensorSupportPack(const Tensor* tensor, bool flag);

    /** 设置张量的填充 */
    static void setTensorPad(const Tensor* tensor, int left, int right, int bottom, int top);
    
    /** 设置共享内存 */
    static void setSharedMem(const Tensor* tensor, Backend::MemObj *mem);
    
    /** 获取共享内存 */
    static Backend::MemObj* getSharedMem(const Tensor* tensor);
};
} // namespace MNN

#endif /* TensorDescribe_hpp */
