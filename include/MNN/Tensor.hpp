//
//  Tensor.hpp
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 * @file Tensor.hpp
 * @brief MNN 张量类
 * 
 * Tensor 是 MNN 中表示多维数组的核心类，提供了以下功能：
 * - 支持多种数据类型（float、int8、int16、int32、uint8等）
 * - 支持多种内存布局（NCHW、NHWC、NC4HW4等）
 * - 支持 CPU 和 GPU 内存管理
 * - 提供方便的形状操作和数据访问接口
 * - 支持跨设备数据传输（CPU ↔ GPU）
 * 
 * 核心特点：
 * - 基于 HalideRuntime.h 中的 halide_buffer_t 实现
 * - 提供 RAII 风格的内存管理
 * - 支持动态形状调整
 * - 提供设备内存映射功能
 */

#ifndef MNN_Tensor_hpp
#define MNN_Tensor_hpp

#include <vector>
#include <MNN/HalideRuntime.h>
#include <MNN/MNNDefine.h>

namespace MNN {

/**
 * data container.
 * data for host tensor is saved in `host` field. its memory is allocated malloc directly.
 * data for device tensor is saved in `deviceId` field. its memory is allocated by session's backend.
 * usually, device tensors are created by engine (like net, session).
 * meanwhile, host tensors could be created by engine or user.
 */
class MNN_PUBLIC Tensor {
public:
    struct InsideDescribe;

    /** 
     * @brief 维度类型枚举
     * 
     * 定义不同的内存布局格式：
     * - TENSORFLOW: NHWC 格式（批量-高度-宽度-通道）
     * - CAFFE: NCHW 格式（批量-通道-高度-宽度）
     * - CAFFE_C4: NC4HW4 格式（4通道打包，适合ARM NEON优化）
     */
    /** dimension type used to create tensor */
    enum DimensionType {
        /** for tensorflow net type. uses NHWC as data format. */
        TENSORFLOW,
        /** for caffe net type. uses NCHW as data format. */
        CAFFE,
        /** for caffe net type. uses NC4HW4 as data format. */
        CAFFE_C4
    };

    /** 
     * @brief 句柄数据类型枚举
     * 
     * 当数据类型为 halide_type_handle 时使用：
     * - HANDLE_NONE: 默认句柄类型
     * - HANDLE_STRING: 字符串句柄类型
     */
    /** handle type */
    enum HandleDataType {
        /** default handle type */
        HANDLE_NONE = 0,
        /** string handle type */
        HANDLE_STRING = 1
    };

    /** 
     * @brief 张量映射类型枚举
     * 
     * 用于 GPU 张量的映射操作：
     * - MAP_TENSOR_WRITE: 映射用于写入数据
     * - MAP_TENSOR_READ: 映射用于读取数据
     */
    /** Tensor map type : Read or Write*/
    enum MapType {
        /** map Tensor for writing data*/
        MAP_TENSOR_WRITE = 0,
        /** map Tensor for reading data*/
        MAP_TENSOR_READ = 1
    };

public:
    /**
     * @brief 创建指定维度大小和类型的张量（不分配内存）
     * @param dimSize 维度数量
     * @param type 维度类型（TENSORFLOW/NCHW/NC4HW4）
     * 
     * 此构造函数创建一个未分配内存的张量，需要后续通过 backend 的 onAcquireBuffer 分配内存。
     */
    /**
     * @brief create a tensor with dimension size and type without acquire memory for data.
     * @param dimSize   dimension size.
     * @param type      dimension type.
     */
    Tensor(int dimSize = 4, DimensionType type = CAFFE);

    /**
     * @brief 创建与指定张量形状相同的张量
     * @param tensor 形状提供者
     * @param type 维度类型
     * @param allocMemory 是否分配内存
     * @warning 张量数据不会被复制
     * 
     * 此构造函数创建一个新的张量，其形状与传入的张量相同。
     */
    /**
     * @brief create a tensor with same shape as given tensor.
     * @param tensor        shape provider.
     * @param type          dimension type.
     * @param allocMemory   acquire memory for data or not.
     * @warning tensor data won't be copied.
     */
    Tensor(const Tensor* tensor, DimensionType type = CAFFE, bool allocMemory = true);

    /** deinitializer */
    ~Tensor();

private:
    Tensor(bool deepCopy, const Tensor* tensor);
    // remove all assignment operator
    Tensor(const Tensor& tensor)  = delete;
    Tensor(const Tensor&& tensor) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(const Tensor&&) = delete;

public:
    /**
     * @brief 创建设备张量（GPU内存）
     * @param shape 张量形状
     * @param type 数据类型
     * @param dimType 维度类型
     * @return 创建的张量
     * @warning 不会分配内存，需要调用 backend 的 onAcquireBuffer 准备内存
     * 
     * 用于创建需要在 GPU 等设备上执行的张量。
     */
    /**
     * @brief create tensor with shape, data type and dimension type.
     * @param shape     tensor shape.
     * @param type      data type.
     * @param dimType   dimension type.
     * @return created tensor.
     * @warning memory for data won't be acquired. call backend's onAcquireBuffer to get memory ready.
     */
    static Tensor* createDevice(const std::vector<int>& shape, halide_type_t type, DimensionType dimType = TENSORFLOW);

    /**
     * @brief 创建设备张量（模板版本）
     * @tparam T 数据类型
     * @param shape 张量形状
     * @param dimType 维度类型
     * @return 创建的张量
     * @warning 不会分配内存，需要调用 backend 的 onAcquireBuffer 准备内存
     * 
     * 使用模板参数自动推断数据类型。
     */
    /**
     * @brief create tensor with shape and dimension type. data type is represented by `T`.
     * @param shape     tensor shape.
     * @param dimType   dimension type.
     * @return created tensor.
     * @warning memory for data won't be acquired. call backend's onAcquireBuffer to get memory ready.
     */
    template <typename T>
    static Tensor* createDevice(const std::vector<int>& shape, DimensionType dimType = TENSORFLOW) {
        return createDevice(shape, halide_type_of<T>(), dimType);
    }

    /**
     * @brief 创建主机张量（CPU内存）
     * @param shape 张量形状
     * @param type 数据类型
     * @param data 数据指针（可选）
     * @param dimType 维度类型
     * @return 创建的张量
     * 
     * 用于创建在 CPU 上执行的张量，可以指定外部数据指针。
     */
    /**
     * @brief create tensor with shape, data type, data and dimension type.
     * @param shape     tensor shape.
     * @param type      data type.
     * @param data      data to save.
     * @param dimType   dimension type.
     * @return created tensor.
     */
    static Tensor* create(const std::vector<int>& shape, halide_type_t type, void* data = NULL,
                          DimensionType dimType = TENSORFLOW);

    /**
     * @brief 创建主机张量（模板版本）
     * @tparam T 数据类型
     * @param shape 张量形状
     * @param data 数据指针（可选）
     * @param dimType 维度类型
     * @return 创建的张量
     * 
     * 使用模板参数自动推断数据类型。
     */
    /**
     * @brief create tensor with shape, data and dimension type. data type is represented by `T`.
     * @param shape     tensor shape.
     * @param data      data to save.
     * @param dimType   dimension type.
     * @return created tensor.
     */
    template <typename T>
    static Tensor* create(const std::vector<int>& shape, void* data = NULL, DimensionType dimType = TENSORFLOW) {
        return create(shape, halide_type_of<T>(), data, dimType);
    }

    /**
     * @brief 克隆张量
     * @param src 源张量
     * @param deepCopy 是否深拷贝（当前只支持 false）
     * @return 克隆的张量
     * 
     * 创建一个新的张量，其形状和类型与源张量相同。
     */
    /**
     * @brief copy tensor.
     * @param src     tensor
     * @param deepCopy whether create new content and copy, currently only support deepCopy = false
     */
    static Tensor* clone(const Tensor* src, bool deepCopy = false);

    /**
     * @brief 销毁张量
     * @param src 要销毁的张量
     * 
     * 释放张量占用的内存和资源。
     */
    /**
     * @brief delete tensor.
     * @param src     tensor
     */
    static void destroy(Tensor* tensor);
public:
    /**
     * @brief 从主机张量复制数据到设备张量
     * @param hostTensor 主机张量（数据提供者）
     * @return 对于设备张量返回 true，对于主机张量返回 false
     * 
     * 用于将 CPU 内存中的数据复制到 GPU 等设备内存。
     */
    /**
     * @brief for DEVICE tensor, copy data from given host tensor.
     * @param hostTensor    host tensor, the data provider.
     * @return true for DEVICE tensor, and false for HOST tensor.
     */
    bool copyFromHostTensor(const Tensor* hostTensor);

    /**
     * @brief 从设备张量复制数据到主机张量
     * @param hostTensor 主机张量（数据消费者）
     * @return 对于设备张量返回 true，对于主机张量返回 false
     * 
     * 用于将 GPU 等设备内存中的数据复制到 CPU 内存。
     */
    /**
     * @brief for DEVICE tensor, copy data to given host tensor.
     * @param hostTensor    host tensor, the data consumer.
     * @return true for DEVICE tensor, and false for HOST tensor.
     */
    bool copyToHostTensor(Tensor* hostTensor) const;

    /**
     * @brief 从设备张量创建主机张量
     * @param deviceTensor 设备张量
     * @param copyData 是否复制数据
     * @return 创建的主机张量
     * 
     * 用于将设备张量转换为主机张量，便于 CPU 访问数据。
     */
    /**
     * @brief create HOST tensor from DEVICE tensor, with or without data copying.
     * @param deviceTensor  given device tensor.
     * @param copyData      copy data or not.
     * @return created host tensor.
     */
    static Tensor* createHostTensorFromDevice(const Tensor* deviceTensor, bool copyData = true);

public:
    /**
     * @brief 获取底层的 halide_buffer_t 结构（常量版本）
     * @return halide_buffer_t 引用
     * 
     * 用于底层操作，不建议直接修改。
     */
    const halide_buffer_t& buffer() const {
        return mBuffer;
    }
    /**
     * @brief 获取底层的 halide_buffer_t 结构（可修改版本）
     * @return halide_buffer_t 引用
     * 
     * 用于底层操作，需要谨慎修改。
     */
    halide_buffer_t& buffer() {
        return mBuffer;
    }

    /**
     * @brief 获取维度类型
     * @return 维度类型
     * 
     * 返回张量的内存布局类型（TENSORFLOW/CAFFE/CAFFE_C4）。
     */
    /**
     * @brief get dimension type.
     * @return dimension type.
     */
    DimensionType getDimensionType() const;

    /**
     * @brief 获取句柄数据类型
     * @return 句柄数据类型
     * 
     * 当数据类型为 halide_type_handle 时使用。
     */
    /**
     * @brief handle data type. used when data type code is halide_type_handle.
     * @return handle data type.
     */
    HandleDataType getHandleDataType() const;

    /**
     * @brief 设置数据类型
     * @param type 数据类型（定义在 Type_generated.h 中）
     * 
     * 用于修改张量的数据类型。
     */
    /**
     * @brief set data type.
     * @param type data type defined in 'Type_generated.h'.
     */
    void setType(int type);

    /**
     * @brief 获取数据类型
     * @return 数据类型
     * 
     * 返回张量的 halide_type_t 类型。
     */
    /**
     * @brief get data type.
     * @return data type.
     */
    inline halide_type_t getType() const {
        return mBuffer.type;
    }

    /**
     * @brief 访问主机内存
     * @tparam T 数据类型
     * @return 类型 T 的数据指针
     * 
     * 用于直接访问 CPU 内存中的数据。
     */
    /**
     * @brief visit host memory, data type is represented by `T`.
     * @return data point in `T` type.
     */
    template <typename T>
    T* host() const {
        return (T*)mBuffer.host;
    }

    /**
     * @brief 访问设备内存
     * @return 设备数据 ID
     * 
     * 设备 ID 的具体含义因后端而异。
     */
    /**
     * @brief visit device memory.
     * @return device data ID. what the ID means varies between backends.
     */
    uint64_t deviceId() const {
        return mBuffer.device;
    }

public:
    /**
     * @brief 获取维度数量
     * @return 维度数量
     * 
     * 返回张量的维度数（如 4 表示 NCHW）。
     */
    int dimensions() const {
        return mBuffer.dimensions;
    }

    /**
     * @brief 获取所有维度的大小
     * @return 维度大小向量
     * 
     * 返回张量各维度的大小，顺序与维度顺序一致。
     */
    /**
     * @brief get all dimensions' extent.
     * @return dimensions' extent.
     */
    std::vector<int> shape() const;

    /**
     * @brief 计算存储数据所需的字节数（考虑重排序标志）
     * @return 所需字节数
     * 
     * 计算张量数据在内存中占用的总字节数。
     */
    /**
     * @brief calculate number of bytes needed to store data taking reordering flag into account.
     * @return bytes needed to store data
     */
    int size() const;
    size_t usize() const;

    /**
     * @brief 计算存储数据所需的元素个数（考虑重排序标志）
     * @return 所需元素个数
     * 
     * 计算张量包含的总元素数。
     */
    /**
     * @brief calculate number of elements needed to store data taking reordering flag into account.
     * @return elements needed to store data
     */
    inline int elementSize() const {
        return size() / mBuffer.type.bytes();
    }

public:
    /**
     * @brief 获取宽度（根据维度类型自动调整）
     * @return 宽度值
     * 
     * - TENSORFLOW 格式：返回 dim[2].extent
     * - CAFFE/CAFFE_C4 格式：返回 dim[3].extent
     */
    inline int width() const {
        if (getDimensionType() == TENSORFLOW) {
            return mBuffer.dim[2].extent;
        }

        return mBuffer.dim[3].extent;
    }
    /**
     * @brief 获取高度（根据维度类型自动调整）
     * @return 高度值
     * 
     * - TENSORFLOW 格式：返回 dim[1].extent
     * - CAFFE/CAFFE_C4 格式：返回 dim[2].extent
     */
    inline int height() const {
        if (getDimensionType() == TENSORFLOW) {
            return mBuffer.dim[1].extent;
        }
        return mBuffer.dim[2].extent;
    }
    /**
     * @brief 获取通道数（根据维度类型自动调整）
     * @return 通道数
     * 
     * - TENSORFLOW 格式：返回 dim[3].extent
     * - CAFFE/CAFFE_C4 格式：返回 dim[1].extent
     */
    inline int channel() const {
        if (getDimensionType() == TENSORFLOW) {
            return mBuffer.dim[3].extent;
        }
        return mBuffer.dim[1].extent;
    }
    /**
     * @brief 获取批量大小
     * @return 批量大小
     * 
     * 返回 dim[0].extent，即批量维度的大小。
     */
    inline int batch() const {
        return mBuffer.dim[0].extent;
    }

    // visit dimension's extent & stride
    /**
     * @brief 获取指定维度的步长
     * @param index 维度索引
     * @return 步长值
     * 
     * 步长是指在该维度上移动一个单位时，内存地址的偏移量。
     */
    inline int stride(int index) const {
        return mBuffer.dim[index].stride;
    }
    /**
     * @brief 获取指定维度的大小
     * @param index 维度索引
     * @return 大小值
     * 
     * 返回指定维度的元素个数。
     */
    inline int length(int index) const {
        return mBuffer.dim[index].extent;
    }
    /**
     * @brief 设置指定维度的步长
     * @param index 维度索引
     * @param stride 步长值
     * 
     * 用于手动调整内存布局。
     */
    inline void setStride(int index, int stride) {
        mBuffer.dim[index].stride = stride;
    }
    /**
     * @brief 设置指定维度的大小
     * @param index 维度索引
     * @param length 大小值
     * 
     * 用于调整张量形状。
     */
    inline void setLength(int index, int length) {
        mBuffer.dim[index].extent = length;
    }

    /**
     * @brief 对于 GPU 和其他设备，直接获取内存信息
     * @param dst 目标内存指针
     * @param forwardType 后端类型
     * @return 是否成功
     * 
     * 如果类型与张量的后端类型不匹配或类型是 CPU，则返回 false。
     */
    /**
     * @brief For GPU and Other Device, get memory directly, see MNNSharedContext for detail
     * @return Success or not. If type != tensor's backend's type or type is cpu , return false
     */
    bool getDeviceInfo(void* dst, int forwardType) const;

public:
    /**
     * @brief 打印张量数据（仅用于调试）
     * 
     * 输出张量的所有元素值，仅用于调试目的。
     */
    /**
     * @brief print tensor data. for DEBUG use only.
     */
    void print() const;

    /**
     * @brief 打印张量形状
     * 
     * 输出张量的形状信息。
     */
    /**
     *@brief print tensor shape
     */
    void printShape() const;

public:
    /**
     * @brief 映射/取消映射 GPU 张量，获取主机指针
     * 
     * 用于在 CPU 端访问 GPU 内存。
     */
    /**
     * @brief map/umap GPU Tensor, to get host ptr
     */
    void* map(MapType mtype, DimensionType dtype);
    void unmap(MapType mtype, DimensionType dtype, void* mapPtr);
    /**
     * @brief 等待直到张量准备好读/写
     * @param mtype 等待读或写
     * @param finish 等待命令刷新或完成
     * 
     * 用于同步 GPU 操作。
     */
    /**
     * @brief wait until the tensor is ready to read / write
     * @param mtype wait for read or write
     * @param finish wait for command flush or finish
     */
    int wait(MapType mtype, bool finish);
    /**
     * @brief 设置 GPU 张量设备指针，并通知内存类型
     * 
     * 用于手动管理 GPU 内存。
     */
    /**
     * @brief set GPU tensor device ptr, and inform memory type
     */
    bool setDevicePtr(const void* devicePtr, int memoryType);
private:
    /** 
     * @brief 底层的 halide_buffer_t 结构
     * 
     * 存储张量的实际数据和元数据。
     */
    halide_buffer_t mBuffer;
    /** 
     * @brief 内部描述结构
     * 
     * 存储额外的张量信息。
     */
    struct InsideDescribe* mDescribe;

private:
    friend class TensorUtils;
};
} // namespace MNN

#endif /* Tensor_hpp */
