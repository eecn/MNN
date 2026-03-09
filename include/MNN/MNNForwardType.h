//
//  MNNForwardType.h
//  MNN
//
//  Created by MNN on 2019/01/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 * @file MNNForwardType.h
 * @brief MNN 后端类型和配置定义
 * 
 * 该文件定义了 MNN 支持的各种后端类型、GPU 模式、后端配置以及运行时状态。
 * 主要包括：
 * - MNNForwardType：后端类型枚举
 * - MNNGpuMode：GPU 调优模式
 * - BackendConfig：后端配置结构体
 * - RuntimeStatus：运行时状态枚举
 * 
 * 这些定义用于指定 MNN 在执行推理时使用的后端，以及配置后端的行为。
 */

#ifndef MNNForwardType_h
#define MNNForwardType_h
#include <stdint.h>
#include <stddef.h>

/**
 * @enum MNNForwardType
 * @brief MNN 后端类型枚举
 * 
 * 定义了 MNN 支持的各种计算后端，用于指定推理时使用的硬件加速方式。
 */
typedef enum {
    /** @brief CPU 后端，默认选项 */
    MNN_FORWARD_CPU = 0,

    /*
     Firtly find the first available backends not equal to CPU
     If no other backends, use cpu
     */
    /** @brief 自动选择后端，优先使用非 CPU 后端 */
    MNN_FORWARD_AUTO = 4,

    /*Hand write metal*/
    /** @brief Metal 后端，用于 Apple 设备的 GPU 加速 */
    MNN_FORWARD_METAL = 1,

    /*NVIDIA GPU API*/
    /** @brief CUDA 后端，用于 NVIDIA GPU 加速 */
    MNN_FORWARD_CUDA = 2,

    /*Android / Common Device GPU API*/
    /** @brief OpenCL 后端，用于支持 OpenCL 的设备 */
    MNN_FORWARD_OPENCL = 3,
    /** @brief OpenGL 后端，用于支持 OpenGL 的设备 */
    MNN_FORWARD_OPENGL = 6,
    /** @brief Vulkan 后端，用于支持 Vulkan 的设备 */
    MNN_FORWARD_VULKAN = 7,

    /*Android 8.1's NNAPI or CoreML for ios*/
    /** @brief NN 后端，包括 Android NNAPI 和 iOS CoreML */
    MNN_FORWARD_NN = 5,

    /*User can use API from Backend.hpp to add or search Backend*/
    /** @brief 用户自定义后端 0 */
    MNN_FORWARD_USER_0 = 8,
    /** @brief 用户自定义后端 1 */
    MNN_FORWARD_USER_1 = 9,
    /** @brief 用户自定义后端 2 */
    MNN_FORWARD_USER_2 = 10,
    /** @brief 用户自定义后端 3 */
    MNN_FORWARD_USER_3 = 11,

    /** @brief 所有后端 */
    MNN_FORWARD_ALL = 12,

    /* Apply arm extension instruction set to accelerate some Ops, this forward type
       is only used in MNN internal, and will be active automatically when user set forward type
       to be MNN_FORWARD_CPU and extension instruction set is valid on hardware.
    */
    /** @brief CPU 扩展后端，使用 ARM 扩展指令集加速 */
    MNN_FORWARD_CPU_EXTENSION = 13,
    // use for shared memory on android device
    
    /** @brief Android 硬件缓冲区内存 */
    MNN_MEMORY_AHARDWAREBUFFER = 14,

    /* For Offline Convert*/
    /** @brief QNN 转换 */
    MNN_CONVERT_QNN = 32,
    /** @brief NEUROPILOT 转换 */
    MNN_CONVERT_NEUROPILOT = 33,
    /** @brief CoreML 转换 */
    MNN_CONVERT_COREML = 34,

} MNNForwardType;

/**
 * @enum MNNGpuMode
 * @brief GPU 调优模式枚举
 * 
 * 定义了 GPU 后端的各种调优选项，用于优化 GPU 执行性能。
 */
typedef enum {
    // For the OpenCL backend, all five of the following options are valid. The user is allowed to enable any one of them.
    // For the Vulkan backend, only options MNN_GPU_TUNING_NONE, MNN_GPU_TUNING_HEAVY, and MNN_GPU_TUNING_WIDE are valid. The user is allowed to enable any one of these three.
    /** @brief 禁止调优，性能较差（OpenCL/Vulkan） */
    MNN_GPU_TUNING_NONE    = 1 << 0,  /* Forbidden tuning, performance not good.(OpenCL/Vulkan) */
    /** @brief 重度调优，通常不建议使用（OpenCL/Vulkan） */
    MNN_GPU_TUNING_HEAVY  = 1 << 1,   /* Heavily tuning, usually not suggested.(OpenCL/Vulkan) */
    /** @brief 广泛调优，性能良好，默认选项（OpenCL/Vulkan） */
    MNN_GPU_TUNING_WIDE   = 1 << 2,   /* Widely tuning, performance good. Default.(OpenCL/Vulkan) */
    /** @brief 普通调优，性能可能不错（仅 OpenCL） */
    MNN_GPU_TUNING_NORMAL = 1 << 3,   /* Normal tuning, performance may be ok.(OpenCL) */
    /** @brief 快速调优，性能可能较差（仅 OpenCL） */
    MNN_GPU_TUNING_FAST   = 1 << 4,   /* Fast tuning, performance may not good.(OpenCL) */

    // For the OpenCL backend, the following two options are both valid. The user could try OpenCL_MEMORY_BUFFER and OpenCL_MEMORY_IMAGE both, and then choose the better one based on performance.
    // For the Vulkan backend, neither option is valid. The user uses the CMake option MNN_VULKAN_IMAGE to select between image memory mode and buffer memory mode.
    /** @brief 使用缓冲区内存模式（仅 OpenCL） */
    MNN_GPU_MEMORY_BUFFER = 1 << 6,   /* OpenCL_MEMORY_BUFFER */
    /** @brief 使用图像内存模式（仅 OpenCL） */
    MNN_GPU_MEMORY_IMAGE  = 1 << 7,   /* OpenCL_MEMORY_IMAGE */

    // For the OpenCL backend, the following two options are effective only on Qualcomm GPUs. When using a Qualcomm GPU, the user could try both options and choose the better one based on performance.
    // For the Vulkan backend, only option MNN_GPU_RECORD_BATCH is valid. When MNN_GPU_RECORD_BATCH is enabled, all ops would share one commandBuffer.
    /** @brief 每个操作记录为一个命令（仅 OpenCL） */
    MNN_GPU_RECORD_OP  = 1 << 8,      /* The kernels in one op execution record into one recording.(OpenCL) */
    /** @brief 批量记录命令（OpenCL：10个内核一个记录；Vulkan：所有操作共享一个命令缓冲区） */
    MNN_GPU_RECORD_BATCH  = 1 << 9,   /* 10 kernels record into one recording.(OpenCL) All ops share one commandBuffer.(Vulkan) */
} MNNGpuMode;

#ifdef __cplusplus
namespace MNN {
/**
 * @struct BackendConfig
 * @brief 后端配置结构体
 * 
 * 用于配置后端的内存、功耗和精度模式，以及设置共享上下文。
 */
struct BackendConfig {
    /**
     * @enum MemoryMode
     * @brief 内存模式枚举
     */
    enum MemoryMode { 
        Memory_Normal = 0,  ///< 普通内存模式
        Memory_High,        ///< 高内存模式（使用更多内存以提高性能）
        Memory_Low          ///< 低内存模式（使用更少内存，性能可能降低）
    };

    /** @brief 内存模式，默认为 Memory_Normal */
    MemoryMode memory = Memory_Normal;

    /**
     * @enum PowerMode
     * @brief 功耗模式枚举
     */
    enum PowerMode { 
        Power_Normal = 0,  ///< 普通功耗模式
        Power_High,        ///< 高功耗模式（性能优先）
        Power_Low          ///< 低功耗模式（节能优先）
    };

    /** @brief 功耗模式，默认为 Power_Normal */
    PowerMode power = Power_Normal;

    /**
     * @enum PrecisionMode
     * @brief 精度模式枚举
     */
    enum PrecisionMode { 
        Precision_Normal = 0,     ///< 普通精度模式
        Precision_High,           ///< 高精度模式
        Precision_Low,            ///< 低精度模式
        Precision_Low_BF16        ///< 低精度 BF16 模式
    };

    /** @brief 精度模式，默认为 Precision_Normal */
    PrecisionMode precision = Precision_Normal;

    /** user defined context */
    /** @brief 用户定义的上下文 */
    union {
        void* sharedContext = nullptr;  ///< 共享上下文指针
        size_t flags;                   ///< 对 CPU 后端有效的标志
    };
};

    /** acquire runtime status by Runtime::getCurrentStatus with following keys,
    */
    /**
     * @enum RuntimeStatus
     * @brief 运行时状态枚举
     * 
     * 用于通过 Runtime::getCurrentStatus 获取运行时状态。
     */
    enum RuntimeStatus {
        /**
         * get status whether this runtime support 16-bits float point arithmetic
         */
        /** @brief 运行时是否支持 16 位浮点运算 */
        STATUS_SUPPORT_FP16,
        /**
         * get status whether this runtime support dot-product arithmetic
         */
        /** @brief 运行时是否支持点积运算 */
        STATUS_SUPPORT_DOT_PRODUCT,
        /**
         * get status whether this runtime support power-low (means low priority for opencl)
         */
        /** @brief 运行时是否支持低功耗模式（对 OpenCL 表示低优先级） */
        STATUS_SUPPORT_POWER_LOW,
        /**
         * emum total number
         */
        /** @brief 枚举总数 */
        STATUS_COUNT
    };

}; // namespace MNN
#endif
#endif /* MNNForwardType_h */
