//
//  Backend.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Backend_hpp
#define Backend_hpp

#include <MNN/MNNForwardType.h>
#include <MNN/ErrorCode.hpp>
#include <map>
#include "Command.hpp"
#include "NonCopyable.hpp"
#include "BufferAllocator.hpp"
#include <future>
#include <atomic>

namespace MNN {

struct Op;
class Execution;

class Runtime;
class Backend;
/**
 * @brief 运行时提示结构体
 * 用于配置运行时行为的各种参数
 */
struct RuntimeHint {
    // 0: Defer, 1: Eager
    int memoryAllocatorType = 0; ///< 内存分配器类型：0=延迟分配，1=即时分配
    int winogradMemoryUsed = 3;  ///< Winograd算法使用的内存量

    // 0-100, 50 means litter core has 50% capacity of large core
    int cpuDecreaseRate = 50;    ///< CPU降速率：0-100，50表示小核心性能为大核心的50%
    int dynamicQuantOption = 0;  ///< 动态量化选项

    // attentionOption % 8:
    // 0: Do not quantize
    // 1: Q,K: Int8, V: Float
    // 2: Q,K,V: Int8

    // attentionOption / 8:
    // 0: don't use flash attention
    // 1: use flash attention

    int attentionOption = 8;     ///< 注意力机制选项：低3位表示量化方式，高位表示是否使用Flash Attention

    // the kvcache size limit of each layer
    // if the size of kvcache in memory exceeds the limit
    // it will be moved to disk to save memory
    // -1 for no limit
    int kvcacheSizeLimit = -1;   ///< KV缓存大小限制（每层），超过则移至磁盘，-1表示无限制

    // path of the kvcache directory
    std::string kvcacheDirPath = ""; ///< KV缓存目录路径

    // path of the kvcache directory
    std::string prefixcacheDirPath = "prefixcache"; ///< 前缀缓存目录路径

    std::string midMemoryPath;   ///< 中间内存路径
    std::string weightMemoryPath; ///< 权重内存路径
    int mmapFileSize = 1024; // MB ///< 内存映射文件大小（MB）
    int useCachedMmap = 0;       ///< 是否使用缓存的内存映射

    // op encoder number for once commit
    int encorderNumForCommit = 10; ///< 每次提交的算子编码器数量
    int initThreadNumber = 0;      ///< 初始化线程数

    // whether to use Arm sme2 cores when threads>1
    bool useArmSme2Cores = true;   ///< 线程数>1时是否使用ARM SME2核心
#ifdef MNN_DEFAULT_USE_KLEIDIAI
    bool enableKleidiAI = true;    ///< 是否启用KleidiAI库
#else
    bool enableKleidiAI = false;   ///< 是否启用KleidiAI库
#endif
    // Use CPU Ids
    std::vector<int> cpuIds;       ///< 使用的CPU核心ID列表

    // Division ration between SME and NEON when runtime threads>=4
    // Default: 41, which means that in LLM inference,
    // during the Prefill stage the workload
    // per single SME core is six times that of NEON,
    //while during the Decode stage it is the same (1×).
    int divisionRatio = 41;        ///< SME与NEON的负载分配比例，默认41

    int smeCores = 2; // Number of SME cores of the backend, default is 2, if supports sme ///< SME核心数量
};
/** 
 * @brief 抽象后端类
 * 所有计算后端的基类，定义了后端的基本接口
 */
class Backend : public NonCopyable {

public:
    /** 
     * @brief 创建后端所需的信息
     * 包含后端类型、线程数/模式、用户配置等
     */
    struct Info {
        /** forward type. */
        MNNForwardType type = MNN_FORWARD_CPU; ///< 前向计算类型
        /** numThread for CPU . number of threads.  gpuMode for GPU only. tuning/memory Mode setting. */
        union {
            int numThread = 4; ///< CPU：线程数
            int gpuMode;       ///< GPU：调优/内存模式设置
        };
        /** user data. */
        BackendConfig* user = NULL; ///< 用户自定义配置
        /**
         * @brief 执行模式枚举
         */
        enum Mode {
            // The Op will be run in execution->onExecute
            DIRECT = 0,   ///< 直接模式：算子在execution->onExecute中执行

            // The Op will be recorded. Run in onExecuteBegin and Wait in onExecuteEnd
            INDIRECT = 1  ///< 间接模式：算子被记录，在onExecuteBegin中运行，在onExecuteEnd中等待
        };
        Mode mode = DIRECT; ///< 执行模式，默认为直接模式
    };

    /** 
     * @brief 后端缓冲区存储类型
     * 定义了不同的内存管理策略
     */
    enum StorageType {
        /**
         use NOT reusable memory.
         - allocates memory when `onAcquireBuffer` is called.
         - releases memory when `onReleaseBuffer` is called or when the backend is deleted.
         - do NOTHING when `onClearBuffer` is called.
         */
        STATIC, ///< 静态存储：不可复用的内存，获取时分配，释放时或后端删除时释放
        /**
         use reusable memory.
         - allocates or reuses memory when `onAcquireBuffer` is called. prefers reusing.
         - collects memory for reuse when `onReleaseBuffer` is called.
         - releases memory when `onClearBuffer` is called or when the backend is deleted.
         */
        DYNAMIC, ///< 动态存储：可复用的内存，优先复用，释放时收集到池中
        /**
         use NOT reusable memory.
         - allocates memory when `onAcquireBuffer` is called.
         - do NOTHING when `onReleaseBuffer` is called.
         - releases memory when `onClearBuffer` is called or when the backend is deleted.
         */
        DYNAMIC_SEPERATE, ///< 独立动态存储：不可复用，获取时分配，清除时释放

        DYNAMIC_IN_EXECUTION ///< 执行中动态存储：在执行期间动态管理
    };

public:
    /**
     * @brief 构造函数
     * @param type 前向计算类型
     */
    Backend(MNNForwardType type) : mType(type) {
        // nothing to do
    }

    /**
     * @brief 析构函数
     */
    virtual ~Backend() = default;

public:

    /**
     * @brief 为算子创建执行器
     * @param inputs    输入张量列表
     * @param outputs   输出张量列表
     * @param op        给定的算子
     * @return 如果支持该算子，返回创建的执行器；否则返回nullptr
     * 
     * @brief create execution for op with input and output tensors.
     * @param inputs    input tensors.
     * @param outputs   output tensors.
     * @param op        given op.
     * @return created execution if op is supported, nullptr otherwise.
     */
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) = 0;

    /**
     * @brief 调整大小前的回调
     * @brief callback before resize ops.
     */
    virtual void onResizeBegin() {
        // nothing to do
    }
    /**
     * @brief 调整大小后的回调
     * @return 错误码
     * @brief callback after resize ops.
     */
    virtual ErrorCode onResizeEnd() = 0;

    /**
     * @brief 执行前的回调
     * @brief callback before executing ops.
     */
    virtual void onExecuteBegin() const = 0;
    /**
     * @brief 执行后的回调
     * @brief callback after executing ops.
     */
    virtual void onExecuteEnd() const = 0;

    /**
     * @brief 获取运行时对象
     * @return 运行时指针，可能为nullptr
     */
    virtual const Runtime* getRuntime() {
        return nullptr;
    }

    /**
     * @brief 为张量分配缓冲区
     * @param tensor        张量对象
     * @param storageType   存储类型
     * @return 是否成功
     * 
     * @brief allocate buffer of tensor for given storage type.
     * @param tensor        buffer provider.
     * @param storageType   buffer storage type.
     * @return success or not.
     */
    MNN_PUBLIC bool onAcquireBuffer(const Tensor* tensor, StorageType storageType);

    /**
     * @brief 释放张量的缓冲区
     * @param tensor        张量对象
     * @param storageType   存储类型
     * @return 是否成功
     * 
     * @brief release buffer of tensor for given storage type.
     * @param tensor        buffer provider.
     * @param storageType   buffer storage type.
     * @return success or not.
     */
    MNN_PUBLIC bool onReleaseBuffer(const Tensor* tensor, StorageType storageType);

    /**
     * @brief 内存对象类
     * 用于管理内存块的生命周期
     */
    class MemObj : public RefCount {
    public:
        MemObj() {}
        virtual ~ MemObj() {}
        /**
         * @brief 获取内存块
         * @return 内存块
         */
        virtual MemChunk chunk() { return MemChunk(); }
    };
    /**
     * @brief 为张量分配缓冲区（虚函数版本）
     * @param tensor        张量对象
     * @param storageType   存储类型
     * @return 内存对象指针，失败返回nullptr
     * 
     * @brief allocate buffer of tensor for given storage type.
     * @param tensor        buffer provider.
     * @param storageType   buffer storage type.
     * @return MemObj for release, if failed, return nullptr.
     */
    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) = 0;

    /**
     * @brief 选择动态分配器
     * @param index     分配器索引
     * @param maxIndex  最大索引
     * @return 是否成功
     */
    virtual bool onSelectDynamicAllocator(int index, int maxIndex) {
        return false;
    }
    /**
     * @brief 直接从张量获取缓冲区信息
     * @param tensor    张量对象
     * @param dstInfo   目标信息指针
     * @return 是否支持
     * 
     * @brief get buffer from tensor directly
     * @param tensor        buffer provider.
     * @return support or not
     */
    virtual bool onGetTensorInfo(const Tensor* tensor, void* dstInfo) {
        return false;
    }

    /**
     * @brief 清除所有动态缓冲区
     * @return 是否成功
     * @brief clear all dynamic buffers.
     * @return success or not.
     */
    virtual bool onClearBuffer() = 0;

    /**
     * @brief 在张量之间复制缓冲区
     * @param srcTensor 源张量
     * @param dstTensor 目标张量
     * 
     * @brief copy buffer from tensor to tensor.
     * @param srcTensor source buffer provider.
     * @param dstTensor dest buffer provider.
     */
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const = 0;

public:
    /**
     * @brief 获取前向计算类型
     * @return 前向计算类型
     * @brief get forward type.
     * @return forward type.
     */
    inline MNNForwardType type() const {
        return mType;
    }

public:
    /**
     * @brief 映射张量到主机指针（用于GPU张量）
     * @param mtype     映射类型
     * @param dtype     维度类型
     * @param srcTensor 源张量
     * @return 主机指针
     * 
     * @brief get Gpu Tensor map host ptr/ unmap
     */
    virtual void* onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor) {
        return nullptr;
    }

    /**
     * @brief 取消映射张量
     * @param mtype     映射类型
     * @param dtype     维度类型
     * @param dstTensor 目标张量
     * @param mapPtr    映射指针
     * @return 是否成功
     */
    virtual bool onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr) {
        return false;
    }

    /**
     * @brief 同步张量数据
     * @param mtype     映射类型
     * @param toCpu     是否同步到CPU
     * @param dstTensor 目标张量
     * @return 状态码
     */
    virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
        return 0;
    }

public:
    /**
     * @brief 获取元数据指针
     * @return 元数据指针
     */
    void* getMetaPtr() {
        return mMetaPtr;
    }
    /**
     * @brief 设置元数据指针
     * @param ptr 元数据指针
     */
    void setMetaPtr(void* ptr) {
        mMetaPtr = ptr;
    }
    // path of the NPU model directory
    std::string pNPUModelDirPath = "."; ///< NPU模型目录路径

private:
    const MNNForwardType mType; ///< 前向计算类型
    void* mMetaPtr; ///< 元数据指针
};

/** 
 * @brief 运行时类
 * 每个后端都属于一个运行时，运行时管理后端的创建和资源
 */
class Runtime : public NonCopyable {
public:
    /**
     Origin Op -> (Compiler) -> New Op -> Backend
     Default use Compiler_Geometry, Origin Op -> Compiler_Geometry -> Little Op
     For serveral Backend, we can't use Geometry to decompose origin op, then it set Compiler_Origin
     
     编译器类型枚举：
     - Geometry: 使用几何计算分解原始算子
     - Origin: 直接使用原始算子
     - Loop: 使用循环编译器
     */
    enum CompilerType {
        Compiler_Geometry = 0, ///< 几何编译器
        Compiler_Origin = 1,   ///< 原始编译器
        Compiler_Loop = 2,     ///< 循环编译器
    };

    /**
     * @brief 分配器类型枚举
     */
    enum AllocatorType {
        Allocator_Defer = 0, ///< 延迟分配器
        Allocator_Eager = 1, ///< 即时分配器
    };
    
    /**
     * @brief 设置运行时提示
     * @param hint 运行时提示
     */
    void setRuntimeHint(const RuntimeHint& hint) {
        mHint = hint;
    }
    
    /**
     * @brief 获取运行时提示
     * @return 运行时提示
     */
    const RuntimeHint& hint() const {
        return mHint;
    }

    /**
     * @brief 获取编译器类型
     * @return 编译器类型
     */
    virtual CompilerType onGetCompilerType() const {
        return Compiler_Loop;
    }

    /**
     * @brief 析构函数
     */
    virtual ~Runtime() = default;
    
    /**
     * @brief 创建后端
     * @param config 后端配置
     * @param origin 原始后端（用于克隆）
     * @return 创建的后端
     * 
     * @brief create backend
     * @return created backend
     */
    virtual Backend* onCreate(const BackendConfig* config = nullptr, Backend* origin = nullptr) const = 0;

    /**
     * @brief 重置运行时
     * @param numberThread 线程数
     * @param config 后端配置
     * @param full 是否完全重置
     * 
     * @brief reset runtime
     */
    virtual void onReset(int numberThread, const BackendConfig* config, bool full) {
        // Do nothing
    }

    /**
     * @brief 垃圾回收
     * @param level 清理级别：0-100，越大清理越多，越小缓存越多
     * 
     * @brief clear unuseful resource
     * @param level clear level: 0 - 100, bigger mean clear more, smaller mean cache more
     */
    virtual void onGabageCollect(int level) = 0;

    /**
     * @brief 获取内存使用量（MB）
     * @return 内存使用量
     * 
     * @brief Measure the memory it used in MB
     */
    virtual float onGetMemoryInMB() {
        return 0.0f;
    }
    
    /**
     * @brief 设置缓存路径（用于NPU后端，不支持从缓冲区加载）
     * @param path 路径
     * @param mode 模式
     * @return 是否成功
     */
    virtual bool onSetCachePath(const char* path, int mode) {
        return false;
    }

    /**
     * @brief 设置缓存
     * @param buffer 缓存缓冲区（不为nullptr时复制缓存，否则删除缓存）
     * @param size 缓存大小
     * @return 是否成功
     */
    virtual bool onSetCache(const void* buffer, size_t size) {
        //default cache valid, avoid being reset
        return true;
    }

    /**
     * @brief 获取缓存
     * @return 缓存指针和大小
     */
    virtual std::pair<const void*, size_t> onGetCache() {
        return std::make_pair(nullptr, 0);
    }
    
    /**
     * @brief 获取运行时状态
     * @param statusEnum 状态枚举
     * @return 状态值
     */
    virtual int onGetRuntimeStatus(RuntimeStatus statusEnum) const {
        return 0;
    }
    
    /**
     * @brief 检查信息是否匹配
     * @param info 后端信息
     * @return 如果用户设置的信息与运行时匹配，返回true；否则返回false并设置真实信息
     */
    virtual bool onCheckInfo(Backend::Info& info) const {
        return true;
    }
    
    /**
     * @brief 算子信息结构体
     */
    struct OpInfo {
        bool initCostLong;      ///< 初始化是否耗时
        float exeutionCost;     ///< 执行耗时（毫秒）
        float initCost;         ///< 初始化耗时（毫秒）
    };
    
    /**
     * @brief 测量算子执行成本
     * @param inputs    输入张量
     * @param outputs   输出张量
     * @param op        给定算子
     * @param dstInfo   输出信息
     * @return 是否支持该算子
     * 
     * @brief measure the cost for op with input and output tensors.
     * @param inputs    input tensors.
     * @param outputs   output tensors.
     * @param op        given op.
     * @param dstInfo   the Info for write.
     * @return support the op or not;
     */
    virtual bool onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                             const MNN::Op* op, OpInfo& dstInfo) const {
        return true;
    }

    /**
     * @brief 标记算子准备就绪（临时接口，将来可能删除）
     * @param inputs  输入张量
     * @param outputs 输出张量
     * @param op      算子
     */
    virtual void onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const MNN::Op* op) {
        // Do nothing
    }
    
    /**
     * @brief 是否已取消（临时接口，将来可能重构）
     */
    std::atomic_bool mCancelled = ATOMIC_VAR_INIT(false);
    
    /**
     * @brief 是否有异步工作
     * @return 是否有异步工作
     */
    MNN_PUBLIC bool hasAsyncWork() const;
    
    /**
     * @brief 设置异步工作
     * @param future 异步future
     */
    void setAsyncWork(std::future<int>&& future);
    
    /**
     * @brief 等待异步工作完成
     */
    MNN_PUBLIC void waitAsyncWork();

    /**
     * @brief 并发开始
     */
    virtual void onConcurrencyBegin() const {
        // Do nothing
    }
    
    /**
     * @brief 并发结束
     */
    virtual void onConcurrencyEnd() const {
        // Do nothing
    }

    mutable int pCurrentStatus = 0; ///< 当前状态，NO_ERROR
    mutable int pExecutionStatus = 0; ///< 执行状态，NO_ERROR

    // TODO: Move to Backend
    void* pMeta = nullptr; ///< 元数据指针
private:
    std::future<int> mFuture; ///< 异步future
    RuntimeHint mHint; ///< 运行时提示
};

/** 
 * @brief 运行时创建器抽象类
 * 用于注册和创建不同类型的运行时
 */
class RuntimeCreator {
public:
    /**
     * @brief 析构函数
     * @brief deinitializer.
     */
    virtual ~RuntimeCreator() = default;

    /**
     * @brief 创建运行时
     * @param info 后端信息
     * @return 创建的运行时
     */
    virtual Runtime* onCreate(const Backend::Info& info) const = 0;
    
    /**
     * @brief 验证信息是否有效
     * @param info 后端信息
     * @return 是否有效
     * 
     * @brief Turn info to supported.
     * @param info    info to valid.
     * @return success or not
     */
    virtual bool onValid(Backend::Info& info) const {
        info.mode = Backend::Info::DIRECT;
        return true;
    }
    
    /**
     * @brief 获取设备信息
     * @param deviceKey   设备键
     * @param deviceValue 设备值输出
     * @return 是否成功
     */
    virtual bool onGetDeviceInfo(const std::string& deviceKey, std::string& deviceValue) const {
        return false;
    }

    /**
     * @brief 设置量化信息
     * @param op      算子
     * @param inputs  输入张量
     * @param outputs 输出张量
     * @return 是否成功
     */
    virtual bool onSetQuantInfo(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) const {
        return false;
    }
protected:
    /**
     * @brief 构造函数
     * @brief deinitializer.
     */
    RuntimeCreator() = default;
};

/**
 * @brief 获取已注册的后端创建器
 * @param type  前向计算类型
 * @return 后端创建器指针，如果未注册返回nullptr
 * 
 * @brief get registered backend creator for given forward type.
 * @param type  given forward type.
 * @return backend creator pointer if registered, nullptr otherwise.
 */
MNN_PUBLIC const RuntimeCreator* MNNGetExtraRuntimeCreator(MNNForwardType type);

/**
 * @brief 注册后端创建器
 * @param type      前向计算类型
 * @param creator   后端创建器
 * @param needCheck 是否需要检查
 * @return 如果该类型的后端创建器之前未注册，返回true；否则返回false
 * 
 * @brief register backend creator for given forward type.
 * @param type given forward type.
 * @param creator registering backend creator.
 * @return true if backend creator for given forward type was not registered before, false otherwise.
 */
MNN_PUBLIC bool MNNInsertExtraRuntimeCreator(MNNForwardType type, const RuntimeCreator* creator,
                                             bool needCheck = false);

/**
 * @brief CPU缓冲区复制
 * @param srcTensor 源张量
 * @param dstTensor 目标张量
 * @return 是否成功
 */
MNN_PUBLIC bool MNNCPUCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor);
} // namespace MNN

#endif /* Backend_hpp */
