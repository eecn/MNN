//
//  Macro.h
//  MNN
//
//  Created by MNN on 2018/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 * @file Macro.h
 * @brief MNN核心宏定义
 * 
 * 本文件定义了MNN内部常用的工具宏和平台检测宏：
 * - 数学运算宏（最小值、最大值、向上取整、内存对齐）
 * - 数据类型定义（BF16精度损失）
 * - 平台特性检测（NEON、ARMv8.2等）
 * 
 * 这些宏主要用于：
 * - 避免重复代码
 * - 提高代码可读性
 * - 实现跨平台兼容性
 */

#ifndef macro_h
#define macro_h
#include <MNN/MNNDefine.h>

// ==================== 数学运算宏 ====================

/**
 * @brief 计算两个值的最小值
 * @param x 第一个值
 * @param y 第二个值
 * @return 较小的值
 * 
 * 注意：参数会被求值两次，如果参数有副作用需谨慎使用
 */
#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))

/**
 * @brief 计算两个值的最大值
 * @param x 第一个值
 * @param y 第二个值
 * @return 较大的值
 * 
 * 注意：参数会被求值两次，如果参数有副作用需谨慎使用
 */
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))

// ==================== 向上取整和对齐宏 ====================

/**
 * @brief 向上取整除法
 * @param x 被除数
 * @param y 除数
 * @return 向上取整的商
 * 
 * 计算公式：ceil(x / y)
 * 
 * 使用场景：
 * - 计算需要多少个块来容纳指定数量的元素
 * - 内存分配时计算需要的块数
 * 
 * 示例：
 * @code
 * UP_DIV(10, 3) = 4    // ceil(10/3) = 4
 * UP_DIV(9, 3) = 3     // ceil(9/3) = 3
 * UP_DIV(8, 4) = 2     // ceil(8/4) = 2
 * @endcode
 */
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

/**
 * @brief 向上对齐到指定边界
 * @param x 原始值
 * @param y 对齐边界（必须是2的幂次方）
 * @return 对齐后的值
 * 
 * 计算公式：ceil(x / y) * y
 * 
 * 使用场景：
 * - 内存对齐（对齐到4、8、16字节等）
 * - 缓存行对齐（通常对齐到64字节）
 * - SIMD指令要求的数据对齐
 * 
 * 示例：
 * @code
 * ROUND_UP(10, 4) = 12   // 对齐到4的倍数
 * ROUND_UP(8, 4) = 8     // 已经对齐
 * ROUND_UP(15, 8) = 16   // 对齐到8的倍数
 * @endcode
 */
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))

/**
 * @brief 向上对齐到4字节边界
 * @param x 原始值
 * @return 对齐到4的倍数后的值
 * 
 * 使用场景：
 * - NEON指令通常要求4字节对齐
 * - 某些数据结构要求4字节对齐
 */
#define ALIGN_UP4(x) ROUND_UP((x), 4)

/**
 * @brief 向上对齐到8字节边界
 * @param x 原始值
 * @return 对齐到8的倍数后的值
 * 
 * 使用场景：
 * - AVX指令通常要求8字节对齐
 * - double类型通常要求8字节对齐
 * - 某些平台的高速缓存行对齐
 */
#define ALIGN_UP8(x) ROUND_UP((x), 8)

// ==================== 数据精度相关宏 ====================

/**
 * @brief BF16（BFloat16）相对于FP32的最大精度损失
 * 
 * BF16是一种16位浮点格式，相比FP32：
 * - 符号位：1位（相同）
 * - 指数位：8位（相同）
 * - 尾数位：7位（FP32是23位）
 * 
 * 因此BF16的尾数精度损失为：23 - 7 = 16位
 * 
 * 计算过程：
 * - BF16尾数位宽：7位
 * - FP32尾数位宽：23位
 * - 精度差异：16位
 * - 最大精度损失：0xffff / 2^23 ≈ 0.00781
 * 
 * 这个值表示BF16相对于FP32的最大相对误差约为0.78%
 * 
 * 使用场景：
 * - 量化/反量化时的精度评估
 * - 模型压缩时的精度损失预估
 * - 混合精度训练时的误差分析
 */
// fraction length difference is 16bit. calculate the real value, it's about 0.00781
#define F32_BF16_MAX_LOSS ((0xffff * 1.0f ) / ( 1 << 23 ))

// ==================== 平台类型定义 ====================

/**
 * @brief Windows平台ssize_t类型定义
 * 
 * 在Windows平台上，标准库没有定义ssize_t类型。
 * 这里使用SSIZE_T（Windows SDK提供的类型）来定义。
 * 
 * ssize_t是size_t的有符号版本，用于表示可能为负的大小值。
 * 常用于系统调用返回值（如read、write）。
 */
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

// ==================== ARM NEON指令集检测 ====================

/**
 * @brief 检测并定义NEON指令集支持
 * 
 * NEON是ARM架构的SIMD（单指令多数据）指令集，
 * 可以同时处理多个数据，显著提升性能。
 * 
 * 检测条件：
 * 1. 平台是ARM架构（32位或64位）
 * 2. 编译器支持NEON（__ARM_NEON__或__ARM_NEON被定义）
 * 
 * 使用场景：
 * - CPU后端优化（图像处理、矩阵运算等）
 * - 向量化计算
 * - 性能关键路径的优化
 * 
 * 示例：
 * @code
 * #ifdef MNN_USE_NEON
 *     // 使用NEON指令集优化
 *     float32x4_t sum = vaddq_f32(a, b);
 * #else
 *     // 使用普通标量运算
 *     for (int i = 0; i < 4; i++) {
 *         sum[i] = a[i] + b[i];
 *     }
 * #endif
 * @endcode
 */
#ifndef MNN_USE_NEON
#if (__arm__ || __aarch64__) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#define MNN_USE_NEON
#endif
#endif

// ==================== ARMv8.2指令集检测 ====================

/**
 * @brief 检测并定义ARMv8.2架构支持
 * 
 * ARMv8.2是ARMv8架构的扩展版本，新增了：
 * - FP16（半精度浮点）算术指令
 * - Dot Product指令（用于矩阵乘法加速）
 * - 其他性能优化指令
 * 
 * 检测条件：
 * 1. 编译时定义了ENABLE_ARMV82
 * 2. 平台是Android ARM64或通用ARM64
 * 3. 如果是iOS模拟器，则不支持（模拟器不支持ARMv8.2）
 * 
 * 使用场景：
 * - FP16推理加速
 * - BF16推理加速
 * - 矩阵乘法优化（Dot Product指令）
 * 
 * 注意：
 * - iOS模拟器不支持ARMv8.2，会被自动禁用
 * - 需要硬件支持ARMv8.2指令集
 * 
 * 示例：
 * @code
 * #ifdef MNN_USE_ARMV82
 *     // 使用ARMv8.2的FP16指令
 *     float16x4_t result = vadd_f16(a, b);
 * #else
 *     // 回退到FP32运算
 *     float32x4_t result = vaddq_f32(a, b);
 * #endif
 * @endcode
 */
#if defined(ENABLE_ARMV82)
#if defined(MNN_BUILD_FOR_ANDROID) || defined(__aarch64__)
#define MNN_USE_ARMV82
#endif

// iOS模拟器不支持ARMv8.2，需要特殊处理
#if defined(__APPLE__)
#if TARGET_OS_SIMULATOR
#ifdef MNN_USE_ARMV82
#undef MNN_USE_ARMV82
#endif
#endif
#endif

#endif

#endif /* macro_h */
