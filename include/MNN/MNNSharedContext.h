//
//  MNNSharedContext.h
//  MNN
//
//  Created by MNN on 2018/10/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 * @file MNNSharedContext.h
 * @brief MNN 共享上下文定义
 * 
 * 该文件定义了 MNN 中用于不同后端的共享上下文结构，主要包括：
 * - Vulkan 后端上下文
 * - Metal 后端上下文
 * - 用户自定义设备上下文
 * 
 * 这些结构用于在 MNN 内部和用户代码之间传递底层图形 API 的上下文信息，
 * 实现不同后端的资源共享和管理。
 */

#ifndef MNNSharedContext_h
#define MNNSharedContext_h
#include "MNNDefine.h"
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h> /*uint32_t*/

/**
 * @brief Vulkan 后端支持
 * 
 * 当定义了 MNN_VULKAN 宏时，启用 Vulkan 相关的结构定义。
 */
#ifdef MNN_VULKAN

/**
 * @struct MNNVulkanContext
 * @brief Vulkan 上下文结构
 * 
 * 包含 Vulkan 后端所需的核心对象，用于在 MNN 和用户代码之间共享 Vulkan 资源。
 */
struct MNNVulkanContext {
    VkInstance pInstance;           ///< Vulkan 实例
    VkPhysicalDevice pPhysicalDevice; ///< 物理设备
    VkDevice pDevice;               ///< 逻辑设备
    VkQueue pQueue;                 ///< 命令队列
    uint32_t iQueueFamilyIndex;     ///< 队列族索引
};

/**
 * @struct MNNVulkanTensorContent
 * @brief Vulkan 张量内容结构
 * 
 * 描述 Vulkan 后端中张量的内存布局和类型信息。
 */
struct MNNVulkanTensorContent {
    VkBuffer buffer;     ///< Vulkan 缓冲区
    VkDeviceSize size;   ///< 缓冲区大小
    VkDeviceSize offset; ///< 偏移量

    halide_type_t realType; ///< 实际数据类型
    int32_t mask;           ///< 预留字段，未来使用
};

#endif

/**
 * @brief Metal 后端支持
 * 
 * 当定义了 MNN_METAL 宏时，启用 Metal 相关的结构定义。
 */
#ifdef MNN_METAL

/**
 * @struct MNNMetalSharedContext
 * @brief Metal 共享上下文结构
 * 
 * 包含 Metal 后端所需的核心对象，用于在 MNN 和用户代码之间共享 Metal 资源。
 */
struct MNNMetalSharedContext {
    id<MTLDevice> device;                ///< Metal 设备
    id<MTLCommandQueue> queue;           ///< 命令队列
};

/**
 * @struct MNNMetalTensorContent
 * @brief Metal 张量内容结构
 * 
 * 描述 Metal 后端中张量的内存布局和类型信息。
 */
struct MNNMetalTensorContent {
    id<MTLBuffer> buffer;     ///< Metal 缓冲区
    int32_t offset;            ///< 偏移量
    id<MTLTexture> texture;    ///< Metal 纹理（用于图像数据）
    
    halide_type_t type;        ///< 数据类型
    int32_t mask;              ///< 掩码
    int32_t forFuture[8];      ///< 预留字段，未来使用
};

/**
 * @brief 获取 Metal 张量内容
 * @param content 输出参数，用于存储张量内容
 * @param tensor 输入张量
 * @return 操作结果，成功返回 0
 */
MNN_PUBLIC int MNNMetalGetTensorContent(MNNMetalTensorContent* content, void* tensor);
#endif

/**
 * @brief 用户自定义设备支持
 * 
 * 当定义了 MNN_USER_SET_DEVICE 宏时，启用用户自定义设备相关的结构定义。
 */
#ifdef MNN_USER_SET_DEVICE

/**
 * @struct MNNDeviceContext
 * @brief 用户自定义设备上下文结构
 * 
 * 用于用户指定自定义设备的配置信息，支持多 GPU 卡和多设备的场景。
 */
struct MNNDeviceContext {
    // When one gpu card has multi devices, choose which device. set deviceId
    uint32_t deviceId = 0;     ///< 设备 ID，当一个 GPU 卡有多个设备时选择
    // When has multi gpu cards, choose which card. set platformId
    uint32_t platformId = 0;   ///< 平台 ID，当有多个 GPU 卡时选择
    // User set number of gpu cards
    uint32_t platformSize = 0;  ///< 用户设置的 GPU 卡数量
    // User set OpenCL context ptr
    void *contextPtr = nullptr; ///< 用户设置的 OpenCL 上下文指针
};

#endif


#ifdef __cplusplus
}
#endif

#endif /* MNNSharedContext_h */
