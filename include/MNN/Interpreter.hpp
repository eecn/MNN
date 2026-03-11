//
//  Interpreter.hpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Interpreter_hpp
#define MNN_Interpreter_hpp

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <MNN/ErrorCode.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

namespace MNN {

/** 
 * @brief 会话调度配置结构体
 * 用于配置模型执行的各种参数
 */
struct ScheduleConfig {
    /** which tensor should be kept */
    std::vector<std::string> saveTensors; ///< 需要保留的张量名称列表
    /** forward type */
    MNNForwardType type = MNN_FORWARD_CPU; ///< 前向计算类型
    /** CPU:number of threads in parallel , Or GPU: mode setting*/
    union {
        int numThread = 4; ///< CPU:并行线程数
        int mode;          ///< GPU:模式设置
    };

    /** subpath to run */
    struct Path {
        std::vector<std::string> inputs;  ///< 输入节点或张量名称
        std::vector<std::string> outputs; ///< 输出节点或张量名称

        enum Mode {
            /**
             * Op Mode
             * - inputs means the source op, can NOT be empty.
             * - outputs means the sink op, can be empty.
             * The path will start from source op, then flow when encounter the sink op.
             * The sink op will not be compute in this path.
             * 
             * @brief 操作模式
             * - inputs 表示源操作，不能为空
             * - outputs 表示目标操作，可以为空
             * 路径从源操作开始，遇到目标操作时结束
             * 目标操作不会在该路径中计算
             */
            Op = 0,

            /**
             * Tensor Mode
             * - inputs means the inputs tensors, can NOT be empty.
             * - outputs means the outputs tensors, can NOT be empty.
             * It will find the pipeline that compute outputs from inputs.
             * 
             * @brief 张量模式
             * - inputs 表示输入张量，不能为空
             * - outputs 表示输出张量，不能为空
             * 会找到从输入张量计算输出张量的管道
             */
            Tensor = 1
        };

        /** running mode */
        Mode mode = Op; ///< 运行模式
    };
    Path path; ///< 子路径配置

    /** backup backend used to create execution when desinated backend do NOT support any op */
    MNNForwardType backupType = MNN_FORWARD_CPU; ///< 当指定后端不支持某些操作时使用的备用后端

    /** extra backend config */
    BackendConfig* backendConfig = nullptr; ///< 额外的后端配置
};

class Session;
struct Content;
class Tensor;
class Backend;
class Runtime;

/** 
 * @brief 操作符信息类
 * 提供操作符的名称、类型和计算量等信息
 */
class MNN_PUBLIC OperatorInfo {
    struct Info;

public:
    /** Operator's name*/
    const std::string& name() const; ///< 获取操作符名称

    /** Operator's type*/
    const std::string& type() const; ///< 获取操作符类型

    /** Operator's flops, in M*/
    float flops() const; ///< 获取操作符计算量（百万次浮点运算）

protected:
    OperatorInfo();
    ~OperatorInfo();
    Info* mContent; ///< 内部信息指针
};

typedef std::function<bool(const std::vector<Tensor*>&, const std::string& /*opName*/)> TensorCallBack;
typedef std::function<bool(const std::vector<Tensor*>&, const OperatorInfo*)> TensorCallBackWithInfo;
typedef std::pair< std::map<MNNForwardType, std::shared_ptr<Runtime>>,  std::shared_ptr<Runtime>> RuntimeInfo;

/**
 * @brief get mnn version info.
 * @return mnn version string.
 * 
 * @brief 获取MNN版本信息
 * @return MNN版本字符串
 */
MNN_PUBLIC const char* getVersion();

/** 
 * @brief 模型解释器类
 * 网络数据持有者，多个会话可以共享同一个网络
 */
class MNN_PUBLIC Interpreter {
public:
    /**
     * @brief create net from file.
     * @param file  given file.
     * @return created net if success, NULL otherwise.
     * 
     * @brief 从文件创建网络
     * @param file 给定的文件路径
     * @return 成功创建的网络，失败返回NULL
     */
    static Interpreter* createFromFile(const char* file);
    /**
     * @brief create net from buffer.
     * @param buffer    given data buffer.
     * @param size      size of data buffer.
     * @return created net if success, NULL otherwise.
     * 
     * @brief 从缓冲区创建网络
     * @param buffer 给定的数据缓冲区
     * @param size 数据缓冲区的大小
     * @return 成功创建的网络，失败返回NULL
     */
    static Interpreter* createFromBuffer(const void* buffer, size_t size);
    ~Interpreter();

    /**
     * @brief destroy Interpreter
     * @param model    given Interpreter to release.
     * 
     * @brief 销毁解释器
     * @param model 要释放的解释器
     */
    static void destroy(Interpreter* net);

    /**
     * @brief 会话模式枚举
     * 定义了会话的各种运行模式
     */
    enum SessionMode {
        /** About CallBack, Default Session_Debug*/
        /** runSessionWithCallBack is allowed and can get internal op info*/
        Session_Debug = 0, ///< 调试模式：允许使用回调并获取内部操作信息
        /** runSessionWithCallBack is not valid and can't get any info of op in session*/
        Session_Release = 1, ///< 释放模式：不允许使用回调，无法获取会话中操作的信息

        /** About input tenosr, Default Session_Input_Inside*/
        /** The input tensor is alloced by session, input data after session resized*/
        Session_Input_Inside = 2, ///< 内部输入模式：输入张量由会话分配，在会话调整大小后输入数据
        /** The input tensor is alloced by user, set input data before session resize*/
        Session_Input_User = 3, ///< 用户输入模式：输入张量由用户分配，在会话调整大小前设置输入数据

        /** The output tensor depends on session, and can't be separate used*/
        Session_Output_Inside = 4, ///< 内部输出模式：输出张量依赖于会话，不能单独使用
        /** The output tensor can be separated from session*/
        Session_Output_User = 5, ///< 用户输出模式：输出张量可以与会话分离

        /** Try Resize Session when create Session or not, default direct: */
        Session_Resize_Direct = 6, ///< 直接调整大小模式：创建会话时尝试调整大小
        Session_Resize_Defer = 7, ///< 延迟调整大小模式：创建会话时不调整大小

        /** Determine the Execution's forward type is determine by user or auto determine */
        Session_Backend_Fix = 8, // Use the backend user set, when not support use default backend
        Session_Backend_Auto = 9, // Auto Determine the Op type by MNN
        ///< 固定后端模式：使用用户设置的后端，不支持时使用默认后端
        ///< 自动后端模式：由MNN自动确定操作类型

        /** Determine static memory whether recyle in resizeSession or just cache the memory */
        Session_Memory_Collect = 10, // Recycle static memory when session resize in case memory explosion
        Session_Memory_Cache = 11, // Cache the static memory for next forward usage
        ///< 内存收集模式：会话调整大小时回收静态内存，防止内存爆炸
        ///< 内存缓存模式：缓存静态内存供下次前向使用

        /** Determine whether use codegen function */
        Session_Codegen_Disable = 12, // Disable codegen in case extra build codegen cost
        Session_Codegen_Enable = 13, // Enable codegen
        ///< 禁用代码生成模式：禁用代码生成以避免额外的构建成本
        ///< 启用代码生成模式：启用代码生成

        /** Dynamic Reisze Optimization */
        Session_Resize_Check = 14, // Open Trace for resize
        Session_Resize_Fix = 15, // Apply Resize Optimization
        ///< 调整大小检查模式：开启调整大小的跟踪
        ///< 调整大小修复模式：应用调整大小优化

        /** Set for Module's traceOrOptimize API.
         Module_Forward_Seperate:
         when inputs is not empty , Module's onForward will only infer shape and alloc memory.
         when inputs is empty , Module's onForward will only runSession to compute content.
         Default is Module_Forward_Combine
         */
        Module_Forward_Separate = 16, ///< 分离前向模式：当输入不为空时，仅推断形状和分配内存；当输入为空时，仅运行会话计算内容
        Module_Forward_Combine = 17, ///< 组合前向模式：默认模式，同时进行形状推断和计算
    };
    /**
     * @brief The API shoud be called before create session.
     * @param mode      session mode
     * 
     * @brief 设置会话模式
     * 该API应在创建会话前调用
     * @param mode 会话模式
     */
    void setSessionMode(SessionMode mode);

    /**
     * @brief The API shoud be called before create session.
     * If the cache exist, try to load cache from file.
     * After createSession, try to save cache to file.
     * @param cacheFile      cache file name
     * @param keySize        depercerate, for future use.
     * 
     * @brief 设置缓存文件
     * 该API应在创建会话前调用
     * 如果缓存存在，尝试从文件加载缓存
     * 创建会话后，尝试将缓存保存到文件
     * @param cacheFile 缓存文件名
     * @param keySize 废弃参数，供未来使用
     */
    void setCacheFile(const char* cacheFile, size_t keySize = 128);

    /**
     * @brief The API shoud be called before create session.
     * @param file      external data file name
     * @param keySize        depercerate, for future use.
     * 
     * @brief 设置外部文件
     * 该API应在创建会话前调用
     * @param file 外部数据文件名
     * @param flag 废弃参数，供未来使用
     */
    void setExternalFile(const char* file, size_t flag = 128);
    /**
     * @brief The API shoud be called after last resize session.
     * If resize session generate new cache info, try to rewrite cache file.
     * If resize session do not generate any new cache info, just do nothing.
     * @param session    given session
     * @param flag   Protected param, not used now
     * 
     * @brief 更新缓存文件
     * 该API应在最后一次调整会话大小后调用
     * 如果调整会话大小生成了新的缓存信息，尝试重写缓存文件
     * 如果调整会话大小没有生成任何新的缓存信息，则不执行任何操作
     * @param session 给定的会话
     * @param flag 保护参数，当前未使用
     * @return 错误码
     */

    ErrorCode updateCacheFile(Session *session, int flag = 0);

    /**
     * @brief 提示模式枚举
     * 用于设置各种优化和配置选项
     */
    enum HintMode {
        // Max Op number for async tuning
        MAX_TUNING_NUMBER = 0, ///< 异步调优的最大操作数
        // Strictly check model file or not, default 1. if set 0, will not check model file valid/invalid
        STRICT_CHECK_MODEL = 1, ///< 是否严格检查模型文件，默认1。如果设置为0，将不检查模型文件的有效性
        MEM_ALLOCATOR_TYPE = 2, ///< 内存分配器类型
        // Winograd unit candidates count, default 3. if set 0, will use less unit candidates for less memory at the expense of performance.
        WINOGRAD_MEMORY_LEVEL = 3, ///< Winograd单元候选数量，默认3。如果设置为0，将使用更少的单元候选以减少内存，但会牺牲性能

        // Geometry Compute option, default is 0xFFFF
        GEOMETRY_COMPUTE_MASK = 4, ///< 几何计算选项，默认0xFFFF

        // default 0
        // 1: For general convolution, use one scale&zeropoint to quant.
        // 2: use block-quant for input data.
        DYNAMIC_QUANT_OPTIONS = 5, ///< 动态量化选项，默认0。1：对于一般卷积，使用一个缩放和零点进行量化；2：对输入数据使用块量化

        // For Mobile CPU with big-litter core, set decrease rate to let MNN divide task differential by CPU's performance
        // 0-100, 50 means litter core has 50% capacity of large core
        // Default is 50
        CPU_LITTLECORE_DECREASE_RATE = 6, ///< 对于具有大小核心的移动CPU，设置降低率让MNN根据CPU性能分配任务。0-100，50表示小核心具有大核心50%的容量。默认50

        // attentionOption % 8:
        // 0: Do not quantize
        // 1: Q,K: Int8, V: Float
        // 2: Q,K,V: Int8

        // attentionOption / 8:
        // 0: don't use flash attention
        // 1: use flash attention
        ATTENTION_OPTION = 7, ///< 注意力选项。取模8：0不量化，1 Q,K为Int8 V为Float，2 Q,K,V均为Int8。整除8：0不使用flash attention，1使用flash attention

        // size limit of kvcache in memory (for a single layer)
        // if the size of kvcache exceeds the limit, it will be moved to disk
        KVCACHE_SIZE_LIMIT = 8, ///< 内存中kvcache的大小限制（单一层）。如果kvcache大小超过限制，将被移至磁盘
        // Op encoder number for commit
        OP_ENCODER_NUMBER_FOR_COMMIT = 9, ///< 用于提交的操作编码器数量

        // KVCache Info
        KVCACHE_INFO = 10, ///< KVCache信息
        // mmap allocate file size, KB
        MMAP_FILE_SIZE = 11, ///< mmap分配文件大小，KB
        USE_CACHED_MMAP = 12, ///< 使用缓存的mmap

        // Multi-Thread Load module, default is 0 (don't use other Thread)
        INIT_THREAD_NUMBER = 13, ///< 多线程加载模块，默认0（不使用其他线程）

        // Used CPU ids
        CPU_CORE_IDS = 14, ///< 使用的CPU核心ID

        // set CPU threads to use when supports Arm sme2
        CPU_SME2_INSTRUCTIONS = 15, ///< 当支持Arm SME2时设置使用的CPU线程

        // Enable KleidiAI
        CPU_ENABLE_KLEIDIAI = 16, ///< 启用KleidiAI

        // Set CPU SME2 NEON division ratio, default is 41
        CPU_SME2_NEON_DIVISION_RATIO = 17, ///< 设置CPU SME2 NEON分割比率，默认41

        // Set SME cores, default is 2, if supports sme
        CPU_SME_CORES = 18 ///< 设置SME核心，默认2（如果支持SME）
    };

    /**
     * @brief 外部路径类型枚举
     * 定义了各种外部资源的路径类型
     */
    enum ExternalPathType {
        // Path of the kvcache directory
        EXTERNAL_PATH_KVCACHE_DIR = 0, ///< KVCache目录路径

        // Mid Buffer Cache File
        EXTERNAL_FEATUREMAP_DIR = 1, ///< 中间缓冲区缓存文件

        // Weight Buffer Cache File
        EXTERNAL_WEIGHT_DIR = 2, ///< 权重缓冲区缓存文件

        // Path of the NPU Model directory
        EXTERNAL_NPU_FILE_DIR = 3, ///< NPU模型目录路径

        // Path of the kvcache directory
        EXTERNAL_PATH_PREFIXCACHE_DIR = 4, ///< 前缀缓存目录路径

        // Other types ...
    };

    /**
     * @brief 几何计算掩码枚举
     * 定义了几何计算的各种选项
     */
    enum GeometryComputeMask {
        // Support Region Fuse
        GEOMETRCOMPUTEMASK_FUSEREGION = 1 << 0, ///< 支持区域融合

        // Support Region Fuse to input with multi-region, eg: pad + concat
        GEOMETRCOMPUTEMASK_FUSEREGION_MULTI = 1 << 1, ///< 支持多区域输入的区域融合，例如：pad + concat

        // Use loop instead of raster + compute if possible
        GEOMETRCOMPUTEMASK_USELOOP = 1 << 2, ///< 尽可能使用循环而不是光栅+计算

        // Support Geometry Cache, if shape changed, will try recompute, and then run compute if failed
        GEOMETRCOMPUTEMASK_OPENCACHE = 1 << 3, ///< 支持几何缓存，如果形状改变，将尝试重新计算，如果失败则运行计算

        // Full option open mask, for example, if want to close useloop, can set mask as (GEOMETRCOMPUTEMASK_ALL - GEOMETRCOMPUTEMASK_USELOOP)
        GEOMETRCOMPUTEMASK_ALL = 0xFFFF, ///< 完整选项开启掩码，例如，如果要关闭useloop，可以将掩码设置为(GEOMETRCOMPUTEMASK_ALL - GEOMETRCOMPUTEMASK_USELOOP)
    };

    /**
     * @brief The API shoud be called before create session.
     * @param hint      Hint type
     * @param value     Hint value
     * @param size      Hint value size(when use a ptr)
     * 
     * @brief 设置会话提示
     * 该API应在创建会话前调用
     * @param hint 提示类型
     * @param value 提示值
     */
    void setSessionHint(HintMode hint, int value);
    
    /**
     * @brief 设置会话提示（指针版本）
     * 该API应在创建会话前调用
     * @param hint 提示类型
     * @param value 提示值指针
     * @param size 提示值大小
     */
    void setSessionHint(HintMode hint, int* value, size_t size);
public:
    /**
     * @brief create runtimeInfo separately with schedule config.
     * @param configs session schedule configs.
     * 
     * @brief 单独创建运行时信息
     * @param configs 会话调度配置列表
     * @return 运行时信息
     */
    static RuntimeInfo createRuntime(const std::vector<ScheduleConfig>& configs);

    /**
     * @brief create session with schedule config. created session will be managed in net.
     * @param config session schedule config.
     * @return created session if success, NULL otherwise.
     * 
     * @brief 创建会话
     * 使用调度配置创建会话，创建的会话将在网络中管理
     * @param config 会话调度配置
     * @return 成功创建的会话，失败返回NULL
     */
    Session* createSession(const ScheduleConfig& config);

    /**
     * @brief create session with schedule config and user-specified runtime.
     * @param config session schedule config, runtime runtimeInfo used by the created session.
     * @return created session if success, NULL otherwise.
     * 
     * @brief 创建会话（指定运行时）
     * 使用调度配置和用户指定的运行时创建会话
     * @param config 会话调度配置
     * @param runtime 创建的会话使用的运行时信息
     * @return 成功创建的会话，失败返回NULL
     */
    Session* createSession(const ScheduleConfig& config, const RuntimeInfo& runtime);

    /**
     * @brief create multi-path session with schedule configs. created session will be managed in net.
     * @param configs session schedule configs.
     * @return created session if success, NULL otherwise.
     * 
     * @brief 创建多路径会话
     * 使用调度配置列表创建多路径会话，创建的会话将在网络中管理
     * @param configs 会话调度配置列表
     * @return 成功创建的会话，失败返回NULL
     */
    Session* createMultiPathSession(const std::vector<ScheduleConfig>& configs);

    /**
     * @brief create multi-path session with schedule configs and user-specified runtime.
              created session will be managed in net.
     * @param configs session schedule configs.
     * @return created session if success, NULL otherwise.
     * 
     * @brief 创建多路径会话（指定运行时）
     * 使用调度配置列表和用户指定的运行时创建多路径会话，创建的会话将在网络中管理
     * @param configs 会话调度配置列表
     * @param runtime 创建的会话使用的运行时信息
     * @return 成功创建的会话，失败返回NULL
     */
    Session* createMultiPathSession(const std::vector<ScheduleConfig>& configs, const RuntimeInfo& runtime);

    /**
     * @brief release session.
     * @param session   given session.
     * @return true if given session is held by net and is freed.
     * 
     * @brief 释放会话
     * @param session 给定的会话
     * @return 如果给定的会话由网络持有并被释放，返回true
     */
    bool releaseSession(Session* session);

    /**
     * @brief call this function to get tensors ready. output tensor buffer (host or deviceId) should be retrieved
     *        after resize of any input tensor.
     * @param session given session.
     * 
     * @brief 调整会话大小
     * 调用此函数使张量准备就绪。在调整任何输入张量大小后，应获取输出张量缓冲区（主机或设备ID）
     * @param session 给定的会话
     */
    void resizeSession(Session* session);

    /**
     * @brief call this function to get tensors ready. output tensor buffer (host or deviceId) should be retrieved
     *        after resize of any input tensor.
     * @param session given session.
     * @param needRelloc, 1 means need realloc.
     * 
     * @brief 调整会话大小（指定是否需要重新分配）
     * 调用此函数使张量准备就绪。在调整任何输入张量大小后，应获取输出张量缓冲区（主机或设备ID）
     * @param session 给定的会话
     * @param needRelloc 是否需要重新分配，1表示需要重新分配
     */
    void resizeSession(Session* session, int needRelloc);


    /**
     * @brief call this function if don't need resize or create session any more, it will save a few memory that equal
     * to the size of model buffer
     * 
     * @brief 释放模型
     * 如果不再需要调整大小或创建会话，调用此函数将节省与模型缓冲区大小相等的内存
     */
    void releaseModel();

    /**
     * @brief Get the model buffer for user to save
     * @return std::make_pair(modelBuffer, modelSize).
     * @example:
     * std::ofstream output("trainResult.alinn")
     * auto buffer = net->getModelBuffer();
     * output.write((const char*)buffer.first, buffer.second);
     * 
     * @brief 获取模型缓冲区
     * 获取模型缓冲区供用户保存
     * @return 模型缓冲区和大小的配对
     * @example:
     * std::ofstream output("trainResult.alinn")
     * auto buffer = net->getModelBuffer();
     * output.write((const char*)buffer.first, buffer.second);
     */
    std::pair<const void*, size_t> getModelBuffer() const;

    /**
     * @brief Get the model's version info.
     * @return const char* of model's version info like "2.0.0";
     * If model is not loaded or model no version info, return "version info not found".
     * 
     * @brief 获取模型版本信息
     * @return 模型版本信息字符串，如"2.0.0"；如果模型未加载或模型无版本信息，返回"version info not found"
     */
    const char* getModelVersion() const;

    /**
     * @brief update Session's Tensor to model's Const Op
     * @param session   given session.
     * @return result of running.
     * 
     * @brief 更新会话的张量到模型的常量操作
     * @param session 给定的会话
     * @return 运行结果
     */
    ErrorCode updateSessionToModel(Session* session);

    /**
     * @brief run session.
     * @param session   given session.
     * @return result of running.
     * 
     * @brief 运行会话
     * @param session 给定的会话
     * @return 运行结果
     */
    ErrorCode runSession(Session* session) const;

    /*
     * @brief run session.
     * @param session   given session.
     * @param before    callback before each op. return true to run the op; return false to skip the op.
     * @param after     callback after each op. return true to continue running; return false to interrupt the session.
     * @param sync      synchronously wait for finish of execution or not.
     * @return result of running.
     * 
     * @brief 带回调运行会话
     * @param session 给定的会话
     * @param before 每个操作前的回调。返回true运行操作；返回false跳过操作
     * @param end 每个操作后的回调。返回true继续运行；返回false中断会话
     * @param sync 是否同步等待执行完成
     * @return 运行结果
     */
    ErrorCode runSessionWithCallBack(const Session* session, const TensorCallBack& before, const TensorCallBack& end,
                                     bool sync = false) const;

    /*
     * @brief run session.
     * @param session   given session.
     * @param before    callback before each op. return true to run the op; return false to skip the op.
     * @param after     callback after each op. return true to continue running; return false to interrupt the session.
     * @param sync      synchronously wait for finish of execution or not.
     * @return result of running.
     * 
     * @brief 带操作符信息回调运行会话
     * @param session 给定的会话
     * @param before 每个操作前的回调。返回true运行操作；返回false跳过操作
     * @param end 每个操作后的回调。返回true继续运行；返回false中断会话
     * @param sync 是否同步等待执行完成
     * @return 运行结果
     */
    ErrorCode runSessionWithCallBackInfo(const Session* session, const TensorCallBackWithInfo& before,
                                         const TensorCallBackWithInfo& end, bool sync = false) const;

    /**
     * @brief get input tensor for given name.
     * @param session   given session.
     * @param name      given name. if NULL, return first input.
     * @return tensor if found, NULL otherwise.
     * 
     * @brief 获取指定名称的输入张量
     * @param session 给定的会话
     * @param name 给定的名称。如果为NULL，返回第一个输入
     * @return 找到的张量，否则返回NULL
     */
    Tensor* getSessionInput(const Session* session, const char* name);
    /**
     * @brief get output tensor for given name.
     * @param session   given session.
     * @param name      given name. if NULL, return first output.
     * @return tensor if found, NULL otherwise.
     * 
     * @brief 获取指定名称的输出张量
     * @param session 给定的会话
     * @param name 给定的名称。如果为NULL，返回第一个输出
     * @return 找到的张量，否则返回NULL
     */
    Tensor* getSessionOutput(const Session* session, const char* name);

    /**
     * @brief 会话信息代码枚举
     * 定义了获取会话信息的各种代码
     */
    enum SessionInfoCode {
        /** memory session used in MB, float* */
        MEMORY = 0, ///< 会话使用的内存，单位MB，float*

        /** float operation needed in session in M, float* */
        FLOPS = 1, ///< 会话中需要的浮点运算，单位M，float*

        /** Backends in session in M, int*, length >= 1 + number of configs when create session */
        BACKENDS = 2, ///< 会话中的后端，int*，长度 >= 1 + 创建会话时的配置数量

        /** Resize Info, int* , the mean different from API
         Interpreter::getSessionInfo: 0: ready to execute, 1: need malloc, 2: need resize
         RuntimeManager::getInfo: 0: no resize, 1: re-malloc, 2: resize
         */
        RESIZE_STATUS = 3, ///< 调整大小信息，int*。Interpreter::getSessionInfo: 0: 准备执行，1: 需要分配内存，2: 需要调整大小。RuntimeManager::getInfo: 0: 无需调整大小，1: 重新分配，2: 调整大小

        /** Mode / NumberThread, int* */
        THREAD_NUMBER = 4, ///< 模式/线程数，int*

        ALL
    };

    /**
     * @brief get session info
     * @param session   given session.
     * @param code      given info code.
     * @param ptr     given info ptr, see SessionInfoCode for detail
     * @return true if support the code, false otherwise.
     * 
     * @brief 获取会话信息
     * @param session 给定的会话
     * @param code 给定的信息代码
     * @param ptr 给定的信息指针，详见SessionInfoCode
     * @return 如果支持该代码，返回true，否则返回false
     */
    bool getSessionInfo(const Session* session, SessionInfoCode code, void* ptr);

    /**
     * @brief get all output tensors.
     * @param session   given session.
     * @return all output tensors mapped with name.
     * 
     * @brief 获取所有输出张量
     * @param session 给定的会话
     * @return 所有输出张量的名称映射
     */
    const std::map<std::string, Tensor*>& getSessionOutputAll(const Session* session) const;
    /**
     * @brief get all input tensors.
     * @param session   given session.
     * @return all input tensors mapped with name.
     * 
     * @brief 获取所有输入张量
     * @param session 给定的会话
     * @return 所有输入张量的名称映射
     */
    const std::map<std::string, Tensor*>& getSessionInputAll(const Session* session) const;

public:
    /**
     * @brief resize given tensor.
     * @param tensor    given tensor.
     * @param dims      new dims. at most 6 dims.
     * 
     * @brief 调整给定张量的大小
     * @param tensor 给定的张量
     * @param dims 新的维度。最多6个维度
     */
    void resizeTensor(Tensor* tensor, const std::vector<int>& dims);

    /**
     * @brief resize given tensor by nchw.
     * @param batch  / N.
     * @param channel   / C.
     * @param height / H.
     * @param width / W
     * 
     * @brief 按NCHW调整给定张量的大小
     * @param batch / N. 批大小
     * @param channel / C. 通道数
     * @param height / H. 高度
     * @param width / W. 宽度
     */
    void resizeTensor(Tensor* tensor, int batch, int channel, int height, int width);

    /**
     * @brief get backend used to create given tensor.
     * @param session   given session.
     * @param tensor    given tensor.
     * @return backend used to create given tensor, may be NULL.
     * 
     * @brief 获取用于创建给定张量的后端
     * @param session 给定的会话
     * @param tensor 给定的张量
     * @return 用于创建给定张量的后端，可能为NULL
     */
    const Backend* getBackend(const Session* session, const Tensor* tensor) const;

    /**
     * @brief get business code (model identifier).
     * @return business code.
     * 
     * @brief 获取业务代码（模型标识符）
     * @return 业务代码
     */
    const char* bizCode() const;

    /**
     * @brief get model UUID
     * @return Model UUID.
     * 
     * @brief 获取模型UUID
     * @return 模型UUID
     */
    const char* uuid() const;

private:
    static Interpreter* createFromBufferInternal(Content* net, bool enforceAuth);

    Content* mNet = nullptr; ///< 网络内容指针
    Interpreter(Content* net);

    Interpreter(const Interpreter&)  = delete;
    Interpreter(const Interpreter&&) = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter& operator=(const Interpreter&&) = delete;
    void waitSessionFinish(const Session* session) const;
#ifdef MNN_INTERNAL_ENABLED
    void logForRunSession(const Session* session, float time, const char* api) const;
#endif
};
} // namespace MNN

#endif /* Interpreter_hpp */
