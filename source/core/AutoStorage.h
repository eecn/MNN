//
//  AutoStorage.h
//  MNN
//
//  Created by MNN on 2018/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef AutoStorage_h
#define AutoStorage_h

#include <stdint.h>
#include <string.h>
#include "MNNMemoryUtils.h"

namespace MNN {
template <typename T>

/** 
 * @brief 自管理内存存储模板类
 * 提供自动内存分配和释放功能的模板类
 */
class AutoStorage {
public:
    /**
     * @brief default initializer.
     * 
     * @brief 默认构造函数
     * 初始化为空存储
     */
    AutoStorage() {
        mSize = 0;
        mData = NULL;
    }
    /**
     * @brief parameter initializer.
     * @param size  number of elements.
     * 
     * @brief 参数构造函数
     * @param size 元素数量
     * 分配指定数量的元素内存空间
     */
    AutoStorage(int size) {
        mData = (T*)MNNMemoryAllocAlign(sizeof(T) * size, MNN_MEMORY_ALIGN_DEFAULT);
        mSize = size;
    }
    /**
     * @brief deinitializer.
     * 
     * @brief 析构函数
     * 自动释放已分配的内存空间
     */
    ~AutoStorage() {
        if ((NULL != mData) && mRelease) {
            MNNMemoryFreeAlign(mData);
        }
    }

    /**
     * @brief get number of elements.
     * @return number of elements..
     * 
     * @brief 获取元素数量
     * @return 元素数量
     */
    inline int size() const {
        return mSize;
    }

    /**
     * @brief set data with number of elements.
     * @param data  data pointer create with `MNNMemoryAllocAlign`.
     * @param size  number of elements.
     * @warning do NOT call `free` or `MNNMemoryFreeAlign` for data pointer passes in.
     * 
     * @brief 设置数据指针和大小
     * @param data 由 `MNNMemoryAllocAlign` 创建的数据指针
     * @param size 元素数量
     * @warning 不要对传入的data指针调用 `free` 或 `MNNMemoryFreeAlign`
     */
    void set(T* data, int size) {
        if (NULL != mData && mData != data) {
            MNNMemoryFreeAlign(mData);
        }
        mData = data;
        mSize = size;
    }

    /**
     * @brief set data with outter pointer, can decide whether free pointer when destructor
     * @param data  data pointer, malloc by outside
     * @param release  whether free pointer when destructor
     * @warning User should ensure memory length is enough
     * 
     * @brief 设置外部数据指针，可决定析构时是否释放
     * @param data 外部分配的数据指针
     * @param release 析构时是否释放指针
     * @warning 用户需确保内存长度足够
     */
    void set(T* data, bool release) {
        mData = data;
        mRelease = release;
    }
    /**
     * @brief reset data size.
     * @param size  number of elements.
     * @warning writed data won't be kept.
     * 
     * @brief 重置数据大小
     * @param size 元素数量
     * @warning 已写入的数据不会被保留
     */
    void reset(int size) {
        if (NULL != mData) {
            MNNMemoryFreeAlign(mData);
        }
        mData = (T*)MNNMemoryAllocAlign(sizeof(T) * size, MNN_MEMORY_ALIGN_DEFAULT);
        mSize = size;
    }

    /**
     * @brief release allocated data.
     * 
     * @brief 释放已分配的内存
     */
    void release() {
        if (mRelease && NULL != mData) {
            MNNMemoryFreeAlign(mData);
            mData = NULL;
            mSize = 0;
        }
    }

    /**
     * @brief set allocated memory data to 0.
     * 
     * @brief 清零分配内存的数据
     */
    void clear() {
        ::memset(mData, 0, mSize * sizeof(T));
    }

    /**
     * @brief get data pointer.
     * @return data pointer.
     * 
     * @brief 获取数据指针
     * @return 数据指针
     */
    T* get() const {
        return mData;
    }

private:
    T* mData  = NULL;     ///< 数据指针
    int mSize = 0;        ///< 元素数量
    bool mRelease = true; ///< 是否在析构时释放内存
};

/** Auto Release Class*/
/** 
 * @brief 自动释放类模板
 * 析构时自动删除指针所指向的对象
 */
template <typename T>
class AutoRelease {
public:
    /**
     * @brief 构造函数
     * @param d 指针对象
     */
    AutoRelease(T* d = nullptr) {
        mData = d;
    }
    /**
     * @brief 析构函数
     * 自动删除指针所指向的对象
     */
    ~AutoRelease() {
        if (NULL != mData) {
            delete mData;
        }
    }
    AutoRelease(const AutoRelease&)  = delete;
    
    /**
     * @brief 重载->操作符
     * @return 指针对象
     */
    T* operator->() {
        return mData;
    }
    
    /**
     * @brief 重置指针
     * @param d 新的指针对象
     * 先删除原对象，再设置新对象
     */
    void reset(T* d) {
        if (nullptr != mData) {
            delete mData;
        }
        mData = d;
    }
    
    /**
     * @brief 获取指针
     * @return 指针对象
     */
    T* get() {
        return mData;
    }
    
    /**
     * @brief 获取常量指针
     * @return 常量指针对象
     */
    const T* get() const {
        return mData;
    }
private:
    T* mData  = NULL; ///< 数据指针
};


/** 
 * @brief 引用计数基类
 * 提供引用计数功能的基类，用于实现自动内存管理
 */
class RefCount
{
    public:
        /**
         * @brief 增加引用计数
         */
        void addRef() const
        {
            mNum++;
        }
        /**
         * @brief 减少引用计数
         * 当引用计数为0时自动删除自身
         */
        void decRef() const
        {
            --mNum;
            MNN_ASSERT(mNum>=0);
            if (0 >= mNum)
            {
                delete this;
            }
        }
        /**
         * @brief 获取引用计数
         * @return 当前引用计数
         */
        inline int count() const{return mNum;}
    protected:
        /**
         * @brief 构造函数
         * 初始引用计数为1
         */
        RefCount():mNum(1){}
        /**
         * @brief 拷贝构造函数
         * @param f 引用的对象
         */
        RefCount(const RefCount& f):mNum(f.mNum){}
        /**
         * @brief 赋值操作符
         * @param f 引用的对象
         */
        void operator=(const RefCount& f)
        {
            if (this != &f)
            {
                mNum = f.mNum;
            }
        }
        /**
         * @brief 析构函数
         */
        virtual ~RefCount(){}
    private:
        mutable int mNum; ///< 引用计数
};

/**
 * @brief 安全释放指针宏
 * 调用指针的decRef方法减少引用计数
 * @param x 指针对象
 */
#define SAFE_UNREF(x)\
    if (NULL!=(x)) {(x)->decRef();}
/**
 * @brief 安全引用指针宏
 * 调用指针的addRef方法增加引用计数
 * @param x 指针对象
 */
#define SAFE_REF(x)\
    if (NULL!=(x)) (x)->addRef();

/**
 * @brief 安全赋值宏
 * 先增加源指针的引用计数，再减少目标指针的引用计数，最后进行赋值
 * @param dst 目标指针
 * @param src 源指针
 */
#define SAFE_ASSIGN(dst, src) \
    {\
        if (src!=NULL)\
        {\
            src->addRef();\
        }\
        if (dst!=NULL)\
        {\
            dst->decRef();\
        }\
        dst = src;\
    }
/**
 * @brief 共享指针模板类
 * 基于引用计数的智能指针，实现自动内存管理
 */
template <typename T>
class SharedPtr {
    public:
        /**
         * @brief 默认构造函数
         */
        SharedPtr() : mT(NULL) {}
        
        /**
         * @brief 参数构造函数
         * @param obj 指针对象
         */
        SharedPtr(T* obj) : mT(obj) {}
        
        /**
         * @brief 拷贝构造函数
         * @param o 共享指针对象
         */
        SharedPtr(const SharedPtr& o) : mT(o.mT) { SAFE_REF(mT); }
        
        /**
         * @brief 析构函数
         * 自动释放引用计数为0的对象
         */
        ~SharedPtr() { SAFE_UNREF(mT); }

        /**
         * @brief 赋值操作符（SharedPtr类型）
         * @param rp 共享指针对象
         * @return 引用当前对象
         */
        SharedPtr& operator=(const SharedPtr& rp) {
            SAFE_ASSIGN(mT, rp.mT);
            return *this;
        }
        
        /**
         * @brief 赋值操作符（T*类型）
         * @param obj 指针对象
         * @return 引用当前对象
         */
        SharedPtr& operator=(T* obj) {
            SAFE_UNREF(mT);
            mT = obj;
            return *this;
        }

        /**
         * @brief 获取原始指针
         * @return 原始指针
         */
        T* get() const { return mT; }
        
        /**
         * @brief 解引用操作符
         * @return 引用对象
         */
        T& operator*() const { return *mT; }
        
        /**
         * @brief 成员访问操作符
         * @return 指针对象
         */
        T* operator->() const { return mT; }

    private:
        T* mT; ///< 内部指针
};

/** 
 * @brief 缓冲区存储结构体
 * 管理动态分配的缓冲区内存
 */
struct BufferStorage {
    /**
     * @brief 获取缓冲区大小
     * @return 可用缓冲区大小（已分配大小减去偏移量）
     */
    size_t size() const {
        return allocated_size - offset;
    }

    /**
     * @brief 获取缓冲区指针
     * @return 缓冲区起始指针（加上偏移量）
     */
    const uint8_t* buffer() const {
        return storage + offset;
    }
    
    /**
     * @brief 析构函数
     * 自动释放分配的缓冲区内存
     */
    ~ BufferStorage() {
        if (nullptr != storage) {
            delete [] storage;
        }
    }
    
    size_t allocated_size;    ///< 已分配的大小
    size_t offset;            ///< 偏移量
    uint8_t* storage = nullptr; ///< 存储缓冲区指针
};

} // namespace MNN

#endif /* AutoStorage_h */
