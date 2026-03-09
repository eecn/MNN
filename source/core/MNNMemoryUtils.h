//
//  MNNMemoryUtils.h
//  MNN
//
//  Created by MNN on 2018/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNMemoryUtils_h
#define MNNMemoryUtils_h

#include <stdio.h>
#include "core/Macro.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 默认内存对齐大小
 * 定义了内存分配时的默认对齐字节数
 */
#define MNN_MEMORY_ALIGN_DEFAULT 64

/**
 * @brief alloc memory with given size & alignment.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 * 
 * @brief 分配指定大小和对齐方式的内存
 * @param size 内存大小，必须大于0
 * @param align 对齐字节数
 * @return 内存指针
 * @warning 必须使用 `MNNMemoryFreeAlign` 释放返回的指针
 * @sa MNNMemoryFreeAlign
 */
MNN_PUBLIC void* MNNMemoryAllocAlign(size_t size, size_t align);

/**
 * @brief alloc memory with given size & alignment, and fill memory space with 0.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 * 
 * @brief 分配指定大小和对齐方式的内存，并将内存空间填充为0
 * @param size 内存大小，必须大于0
 * @param align 对齐字节数
 * @return 内存指针
 * @warning 必须使用 `MNNMemoryFreeAlign` 释放返回的指针
 * @sa MNNMemoryFreeAlign
 */
MNN_PUBLIC void* MNNMemoryCallocAlign(size_t size, size_t align);

/**
 * @brief free aligned memory pointer.
 * @param mem   aligned memory pointer.
 * @warning do NOT pass any pointer NOT returned by `MNNMemoryAllocAlign` or `MNNMemoryCallocAlign`.
 * @sa MNNMemoryAllocAlign
 * @sa MNNMemoryCallocAlign
 * 
 * @brief 释放对齐的内存指针
 * @param mem 对齐的内存指针
 * @warning 不要传递非 `MNNMemoryAllocAlign` 或 `MNNMemoryCallocAlign` 返回的指针
 * @sa MNNMemoryAllocAlign
 * @sa MNNMemoryCallocAlign
 */
MNN_PUBLIC void MNNMemoryFreeAlign(void* mem);

#ifdef __cplusplus
}
#endif

#endif /* MNNMemoryUtils_h */
