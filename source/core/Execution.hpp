//
//  Execution.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#ifndef Execution_hpp
#define Execution_hpp

#include <MNN/MNNForwardType.h>
#include <MNN/ErrorCode.hpp>
#include <MNN/Tensor.hpp>
#include <memory>
#include <string>
#include "NonCopyable.hpp"

namespace MNN {
class Backend;
struct Op;

/** 抽象执行单元 */
/** abstract execution */
class Execution : public NonCopyable {
public:
    /**
     * @brief 构造函数
     * @param backend   执行单元运行的后端
     */
    /**
     * @brief initializer.
     * @param backend   backend that exection will running on.
     */
    Execution() = delete;
    Execution(Backend *backend) : mBackEnd(backend) {
        // nothing to do
    }
    /**
     * @brief 析构函数
     */
    /**
     * @brief deinitializer.
     */
    virtual ~Execution() = default;

    /**
     * @brief 响应输入或输出张量的形状变化
     * @param inputs    输入张量
     * @param outputs   输出张量
     * @return 调整大小的结果
     */
    /**
     * @brief response shape change of input or output tensors.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return resize result
     */
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        return NO_ERROR;
    }

    /**
     * @brief 执行计算
     * @param inputs    输入张量
     * @param outputs   输出张量
     * @return 执行结果
     */
    /**
     * @brief perform execution.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return execution result
     */
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) = 0;

    /**
     * @brief 克隆执行单元，新执行单元将共享原执行单元的权重
     * @param bn   克隆执行单元的后端
     * @param op   操作符
     * @param dst  如果为nullptr，仅返回是否可以克隆；否则将执行单元克隆到dst
     * @return 是否成功克隆
     */
    /**
     * @brief clone execution, new execution will share weight from this execution
     * @param bn   the cloned' execution's backend
     * @param op   the operation
     * @param dst if dst = nullptr, just return whether execution can clone, otherwise clone the execution into dst
     * @return execution result
     */
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) {
        return false;
    }
public:
    /**
     * @brief 为插件系统设计，尚未就绪
     */
    /**
     * @brief designed for plugin system. not ready yet.
     */
    class Creator : public NonCopyable {
    public:
        /**
         * @brief 析构函数
         */
        /**
         * @brief deinitializer.
         */
        virtual ~Creator() = default;
        /**
         * @brief 为给定的操作和后端创建执行单元
         * @param backend   给定的后端
         * @param op        给定的操作
         * @return 执行单元
         */
        /**
         * @brief create execution for given op on given backend.
         * @param backend   given backend.
         * @param op        given op.
         * @return execution.
         */
        virtual Execution *onCreate(Backend *backend, const Op *op) const = 0;
    };

    // 搜索额外的创建器，如果未找到，返回nullptr
    // Search for extra creator, if not found, return nullptr
    MNN_PUBLIC static const Creator *searchExtraCreator(const std::string &key, MNNForwardType type);

    /**
     * @brief 为给定的键和后端类型注册创建器
     * @param creator 要注册的创建器
     * @param key 给定的键
     * @param type 给定的后端类型
     * @return 如果已存在相同键和类型的注册创建器，返回false；否则返回true
     */
    /**
     * @brief register creator for given key and backend type.
     * @param creator registering creator.
     * @param key given key.
     * @param type given backend type.
     * @return false if registered creator for same key and type exists, true otherwise.
     */
    MNN_PUBLIC static bool insertExtraCreator(std::shared_ptr<Creator> creator, const std::string &key,
                                              MNNForwardType type);

    /**
     * @brief 为给定的键和后端类型注销创建器
     * @param key 给定的键
     * @param type 给定的后端类型
     * @return 如果存在给定键和类型的注册创建器，返回true；否则返回false
     */
    /**
     * @brief unregister creator for given key and backend type.
     * @param key given key.
     * @param type given backend type.
     * @return true if registered creator for given key and type exists, false otherwise.
     */
    MNN_PUBLIC static bool removeExtraCreator(const std::string &key, MNNForwardType type);

public:
    /**
     * @brief 检查执行单元是否有效
     * @return 是否有效
     */
    /**
     * @brief check if execution is valid.
     * @return valid or not.
     */
    inline bool valid() const {
        return mValid;
    }
    /**
     * @brief 执行单元的输入和输出是否需要在外部分配
     * @return 是否需要
     */
    /**
     * @brief the execution's inputs and outputs should will alloc outside
     * @return yes or not.
     */
    inline bool needAllocIO() const {
        return mNeedAllocIO;
    }
    /**
     * @brief 获取后端
     * @return 后端
     */
    /**
     * @brief get backend.
     * @return backend.
     */
    Backend *backend() const {
        return mBackEnd;
    }

protected:
    bool mValid = true;         ///< 执行单元是否有效
    bool mNeedAllocIO = true;    ///< 是否需要在外部分配输入输出

private:
    Backend *mBackEnd;           ///< 执行单元所属的后端
};

} // namespace MNN

#endif /* Execution_hpp */
