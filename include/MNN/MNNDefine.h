//
//  MNNDefine.h
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 * @file MNNDefine.h
 * @brief MNN基础宏定义文件
 * 
 * 本文件定义了MNN库中使用的各种宏，包括：
 * - 平台检测宏
 * - 日志输出宏
 * - 断言宏
 * - DLL导出/导入宏
 * - 版本号定义
 */

#ifndef MNNDefine_h
#define MNNDefine_h

#include <assert.h>
#include <stdio.h>

// ==================== 平台检测宏 ====================

/** 
 * Apple平台检测
 * 如果是iOS平台，定义MNN_BUILD_FOR_IOS宏
 */
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define MNN_BUILD_FOR_IOS
#endif
#endif

// ==================== 日志输出宏 ====================

/** 
 * 日志输出宏定义
 * 根据不同平台使用不同的日志输出方式：
 * - Android/鸿蒙：使用系统日志API
 * - iOS：同时输出到syslog和stderr
 * - 其他平台：使用标准printf
 */
#ifdef MNN_USE_LOGCAT
#if defined(__OHOS__)
// 鸿蒙系统日志输出
#include <hilog/log.h>
#define MNN_ERROR(format, ...) {char logtmp[4096]; snprintf(logtmp, 4096, format, ##__VA_ARGS__); OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, "MNNJNI", (const char*)logtmp);}
#define MNN_PRINT(format, ...) {char logtmp[4096]; snprintf(logtmp, 4096, format, ##__VA_ARGS__); OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_DOMAIN, "MNNJNI", (const char*)logtmp);}
#else
// Android系统日志输出
#include <android/log.h>
#define MNN_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MNNJNI", format, ##__VA_ARGS__)
#define MNN_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#endif
#elif defined MNN_BUILD_FOR_IOS
// on iOS, stderr prints to XCode debug area and syslog prints Console. You need both.
// iOS平台日志输出：同时输出到syslog和stderr，以便在XCode调试区域和控制台都能看到
#include <syslog.h>
#define MNN_PRINT(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#else
// 其他平台使用标准printf输出
#define MNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

// ==================== 断言和调试宏 ====================

/** 
 * MNN断言宏
 * 在DEBUG模式下，如果条件不满足，输出错误信息并触发assert
 * 在非DEBUG模式下，该宏不执行任何操作
 * @param x 需要检查的条件表达式
 */
#ifdef DEBUG
#define MNN_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            MNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define MNN_ASSERT(x)
#endif

/** 
 * 函数打印宏 - 打印整数值
 * 打印变量名、整数值、函数名和行号
 * @param x 要打印的整数变量
 */
#define FUNC_PRINT(x) MNN_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);

/** 
 * 函数打印宏 - 打印任意类型值
 * 打印变量名、类型、值、函数名和行号
 * @param x 要打印的变量
 * @param type 变量的格式化类型（如d、f、s等）
 */
#define FUNC_PRINT_ALL(x, type) MNN_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

/** 
 * 检查宏
 * 如果条件不满足，输出错误信息
 * @param success 需要检查的条件
 * @param log 错误提示信息
 */
#define MNN_CHECK(success, log) \
if(!(success)){ \
MNN_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}

// ==================== DLL导出/导入宏 ====================

/** 
 * MNN_PUBLIC宏 - 控制符号的可见性
 * 在Windows平台：
 * - BUILDING_MNN_DLL定义时：导出符号（__declspec(dllexport)）
 * - USING_MNN_DLL定义时：导入符号（__declspec(dllimport)）
 * - 其他情况：空定义
 * 在非Windows平台：使用__attribute__((visibility("default")))设置符号可见性
 */
#if defined(_MSC_VER)
#if defined(BUILDING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllexport)
#elif defined(USING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllimport)
#else
#define MNN_PUBLIC
#endif
#else
#define MNN_PUBLIC __attribute__((visibility("default")))
#endif

// ==================== 版本号定义 ====================

/** 
 * 字符串化辅助宏
 * 将参数转换为字符串字面量
 */
#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)

/** 
 * MNN版本号定义
 * 遵循语义化版本规范（Semantic Versioning）：MAJOR.MINOR.PATCH
 */
#define MNN_VERSION_MAJOR 3  // 主版本号：不兼容的API修改
#define MNN_VERSION_MINOR 4  // 次版本号：向下兼容的功能性新增
#define MNN_VERSION_PATCH 1  // 修订号：向下兼容的问题修正

/** 
 * MNN完整版本字符串
 * 格式："主版本号.次版本号.修订号"
 */
#define MNN_VERSION STR(MNN_VERSION_MAJOR) "." STR(MNN_VERSION_MINOR) "." STR(MNN_VERSION_PATCH)

#endif /* MNNDefine_h */
