//
//  AutoTime.hpp
//  MNN
//
//  Created by MNN on 2018/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 * @file AutoTime.hpp
 * @brief 高性能计时工具类
 * 
 * 本文件提供了两个主要的计时工具：
 * - Timer: 基础计时器类，支持手动控制计时
 * - AutoTime: 自动计时类，利用RAII机制自动记录代码块执行时间
 * 
 * 使用场景：
 * - 性能分析和瓶颈定位
 * - 代码执行时间统计
 * - 调试时追踪耗时操作
 */

#ifndef MNN_AutoTime_hpp
#define MNN_AutoTime_hpp

#include <stdint.h>
#include <stdio.h>
#include <MNN/MNNDefine.h>

namespace MNN {

/**
 * @brief 基础计时器类
 * 
 * 提供精确的微秒级计时功能。
 * 使用平台特定的高精度计时器实现（如QueryPerformanceCounter on Windows）。
 * 
 * 使用示例：
 * @code
 * MNN::Timer timer;
 * // ... 执行一些操作 ...
 * uint64_t elapsedUs = timer.durationInUs();
 * printf("耗时: %llu 微秒\n", elapsedUs);
 * 
 * timer.reset();  // 重置计时器
 * // ... 再次执行操作 ...
 * elapsedUs = timer.durationInUs();
 * @endcode
 */
class MNN_PUBLIC Timer {
public:
    Timer();
    ~Timer();
    Timer(const Timer&)  = delete;
    Timer(const Timer&&) = delete;
    Timer& operator=(const Timer&)  = delete;
    Timer& operator=(const Timer&&) = delete;
    
    /**
     * @brief 重置计时器
     * 
     * 将计时起点设置为当前时间。
     * 后续调用 durationInUs() 将计算从这次重置到当前的时间差。
     */
    // reset timer
    void reset();
    
    /**
     * @brief 获取经过的时间（微秒）
     * @return 从构造或上次reset()以来经过的微秒数
     * 
     * 注意：1秒 = 1,000,000微秒
     */
    // get duration (us) from init or latest reset.
    uint64_t durationInUs();
    
    /**
     * @brief 获取上次重置的时间点
     * @return 上次reset()时记录的时间戳
     * 
     * 这个时间戳是平台相关的内部表示，
     * 通常用于计算时间差而非直接解读。
     */
    // Get Current Time
    uint64_t current() const {
        return mLastResetTime;
    }
protected:
    uint64_t mLastResetTime;  ///< 上次重置时的时间戳（平台特定的内部表示）
};

/** time tracing util. prints duration between init and deinit. */
/**
 * @brief 自动计时工具类（RAII风格）
 * 
 * 利用C++的RAII（资源获取即初始化）机制，
 * 在对象构造时开始计时，在对象析构时自动输出耗时。
 * 
 * 这是性能分析中最常用的工具，无需手动调用开始/结束。
 * 
 * 使用示例：
 * @code
 * void myFunction() {
 *     AUTOTIME;  // 宏定义，展开为 AutoTime ___t(__LINE__, __func__)
 *     
 *     // ... 函数代码 ...
 *     // 当函数返回时，会自动打印该函数的执行时间
 * }
 * 
 * // 或者手动使用（不推荐，宏更方便）：
 * {
 *     MNN::AutoTime timer(__LINE__, "code_block");
 *     // ... 代码块 ...
 * }  // 代码块结束时自动打印耗时
 * @endcode
 * 
 * @see AUTOTIME 宏定义，提供更简洁的使用方式
 */
 
// time tracing util. prints duration between init and deinit.
class MNN_PUBLIC AutoTime : Timer {
public:
    /**
     * @brief 构造函数，开始计时
     * @param line 代码行号（通常使用 __LINE__ 宏）
     * @param func 函数名（通常使用 __func__ 宏）
     * 
     * 这些信息将用于输出，帮助定位被计时的代码位置。
     */
    AutoTime(int line, const char* func);
    
    /**
     * @brief 析构函数，自动输出耗时
     * 
     * 当AutoTime对象离开作用域被销毁时，
     * 会自动计算并打印从构造到此刻的耗时。
     */
    ~AutoTime();
    
    AutoTime(const AutoTime&)  = delete;
    AutoTime(const AutoTime&&) = delete;
    AutoTime& operator=(const AutoTime&) = delete;
    AutoTime& operator=(const AutoTime&&) = delete;

private:
    int mLine;      ///< 记录代码行号，用于输出定位
    char* mName;    ///< 记录函数名，用于输出标识
};
} // namespace MNN

/**
 * @def AUTOTIME
 * @brief 自动计时的便捷宏
 * 
 * 当定义了 MNN_OPEN_TIME_TRACE 宏时，
 * 此宏展开为创建一个 AutoTime 对象，自动记录当前函数的执行时间。
 * 
 * 当未定义 MNN_OPEN_TIME_TRACE 时，此宏展开为空，不产生任何开销。
 * 
 * 使用方式：
 * @code
 * void myFunction() {
 *     AUTOTIME;  // 放在函数开头即可
 *     // ... 函数代码 ...
 * }
 * @endcode
 * 
 * 输出示例：
 * @code
 * [MNN]: myFunction, 123, cost time: 150 us
 * @endcode
 */
#ifdef MNN_OPEN_TIME_TRACE
#define AUTOTIME MNN::AutoTime ___t(__LINE__, __func__)
#else
#define AUTOTIME
#endif

#endif /* AutoTime_hpp */
