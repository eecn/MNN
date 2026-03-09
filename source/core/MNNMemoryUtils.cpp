//
//  MNNMemoryUtils.cpp
//  MNN
//
//  Created by MNN on 2018/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/MNNMemoryUtils.h"
#include <stdint.h>
#include <stdlib.h>
#include "core/Macro.h"
//#define MNN_DEBUG_MEMORY
/**
 * @brief 指针对齐函数
 * 将指针调整到指定的对齐边界
 * @param ptr 原始指针
 * @param alignment 对齐字节数
 * @return 对齐后的指针
 */
static inline void **alignPointer(void **ptr, size_t alignment) {
    return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
}

/**
 * @brief 分配对齐内存
 * 分配指定大小和对齐方式的内存
 * @param size 内存大小
 * @param alignment 对齐字节数
 * @return 对齐后的内存指针
 */
extern "C" void *MNNMemoryAllocAlign(size_t size, size_t alignment) {
    MNN_ASSERT(size > 0);

#ifdef MNN_DEBUG_MEMORY
    return malloc(size);
#else
    // 分配额外的空间用于存储原始指针和对齐调整
    void **origin = (void **)malloc(size + sizeof(void *) + alignment);
    MNN_ASSERT(origin != NULL);
    if (!origin) {
        return NULL;
    }

    // 计算对齐后的指针位置
    void **aligned = alignPointer(origin + 1, alignment);
    // 存储原始指针，用于后续释放
    aligned[-1]    = origin;
    return aligned;
#endif
}

/**
 * @brief 分配对齐内存并初始化为0
 * 分配指定大小和对齐方式的内存，并将内存空间填充为0
 * @param size 内存大小
 * @param alignment 对齐字节数
 * @return 对齐后的内存指针
 */
extern "C" void *MNNMemoryCallocAlign(size_t size, size_t alignment) {
    MNN_ASSERT(size > 0);

#ifdef MNN_DEBUG_MEMORY
    return calloc(size, 1);
#else
    // 分配额外的空间用于存储原始指针和对齐调整，并初始化为0
    void **origin = (void **)calloc(size + sizeof(void *) + alignment, 1);
    MNN_ASSERT(origin != NULL);
    if (!origin) {
        return NULL;
    }
    // 计算对齐后的指针位置
    void **aligned = alignPointer(origin + 1, alignment);
    // 存储原始指针，用于后续释放
    aligned[-1]    = origin;
    return aligned;
#endif
}

/**
 * @brief 释放对齐内存
 * 释放由 MNNMemoryAllocAlign 或 MNNMemoryCallocAlign 分配的内存
 * @param aligned 对齐后的内存指针
 */
extern "C" void MNNMemoryFreeAlign(void *aligned) {
#ifdef MNN_DEBUG_MEMORY
    free(aligned);
#else
    if (aligned) {
        // 获取原始指针并释放
        void *origin = ((void **)aligned)[-1];
        free(origin);
    }
#endif
}
