//
//  ImageProcess.hpp
//  MNN
//
//  Created by MNN on 2018/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_ImageProcess_hpp
#define MNN_ImageProcess_hpp

#include <MNN/ErrorCode.hpp>
#include <MNN/Matrix.h>
#include <MNN/Tensor.hpp>

namespace MNN {
namespace CV {
/**
 * @brief 图像格式枚举
 * 定义了支持的各种图像像素格式
 */
enum ImageFormat {
    RGBA     = 0,  ///< RGBA格式，4通道
    RGB      = 1,  ///< RGB格式，3通道
    BGR      = 2,  ///< BGR格式，3通道
    GRAY     = 3,  ///< 灰度格式，1通道
    BGRA     = 4,  ///< BGRA格式，4通道
    YCrCb    = 5,  ///< YCrCb格式
    YUV      = 6,  ///< YUV格式
    HSV      = 7,  ///< HSV格式
    XYZ      = 8,  ///< XYZ格式
    BGR555   = 9,  ///< BGR555格式，16位
    BGR565   = 10, ///< BGR565格式，16位
    YUV_NV21 = 11, ///< YUV NV21格式
    YUV_NV12 = 12, ///< YUV NV12格式
    YUV_I420 = 13, ///< YUV I420格式
    HSV_FULL = 14, ///< 完整HSV格式
};

/**
 * @brief 图像插值滤波器枚举
 * 定义了图像缩放时使用的插值方法
 */
enum Filter { 
    NEAREST = 0,  ///< 最近邻插值
    BILINEAR = 1, ///< 双线性插值
    BICUBIC = 2   ///< 双三次插值
};

/**
 * @brief 图像边缘处理方式枚举
 * 定义了处理图像边缘像素的方法
 */
enum Wrap { 
    CLAMP_TO_EDGE = 0, ///< 边缘像素复制
    ZERO = 1,          ///< 边缘填充为0
    REPEAT = 2         ///< 边缘循环复制
 };

/**
 * handle image process for tensor.
 * step:
 *  1: Do transform compute and get points
 *  2: Sample line and do format convert
 *  3: Turn RGBA to float tensor, and do sub and normalize
 */

class MNN_PUBLIC ImageProcess {
public:
    struct Inside;
    /**
     * @brief 图像处理配置结构体
     * 用于设置图像处理的各种参数
     */
    struct Config {
        /** data filter */
        Filter filterType = NEAREST; ///< 插值滤波器类型
        /** format of source data */
        ImageFormat sourceFormat = RGBA; ///< 源图像格式
        /** format of destination data */
        ImageFormat destFormat = RGBA; ///< 目标图像格式

        // Only valid if the dest type is float
        float mean[4]   = {0.0f, 0.0f, 0.0f, 0.0f};   ///< 均值，用于图像归一化
        float normal[4] = {1.0f, 1.0f, 1.0f, 1.0f}; ///< 归一化系数

        /** edge wrapper */
        Wrap wrap = CLAMP_TO_EDGE; ///< 边缘处理方式
    };
public:
    /**
     * @brief create image process with given config for given tensor.
     * @param config    given config.
     * @param dstTensor given tensor.
     * @return image processor.
     * 
     * @brief 创建图像处理对象
     * @param config 配置参数结构体
     * @param dstTensor 目标张量，可选
     * @return 图像处理对象指针
     */
    static ImageProcess* create(const Config& config, const Tensor* dstTensor = nullptr);

    /**
     * @brief create image process with given config for given tensor.
     * @param means given means
     * @param meanCount given means count
     * @param normals   given normals
     * @param normalCount given normal count
     * @param sourceFormat  format of source data
     * @param destFormat    format of destination data
     * @param dstTensor given tensor.
     * @return image processor.
     * 
     * @brief 创建图像处理对象（重载版本）
     * @param sourceFormat 源图像格式
     * @param destFormat 目标图像格式
     * @param means 均值数组
     * @param meanCount 均值数量
     * @param normals 归一化系数数组
     * @param normalCount 归一化系数数量
     * @param dstTensor 目标张量，可选
     * @return 图像处理对象指针
     */
    static ImageProcess* create(const ImageFormat sourceFormat = RGBA, const ImageFormat destFormat = RGBA,
                                const float* means = nullptr, const int meanCount = 0, const float* normals = nullptr,
                                const int normalCount = 0, const Tensor* dstTensor = nullptr);
    
    ~ImageProcess();
    static void destroy(ImageProcess* imageProcess);
    
    /**
     * @brief get affine transform matrix.
     * @return affine transform matrix.
     * 
     * @brief 获取仿射变换矩阵
     * @return 仿射变换矩阵的常量引用
     */
    inline const Matrix& matrix() const {
        return mTransform;
    }
    
    /**
     * @brief 设置仿射变换矩阵
     * @param matrix 要设置的仿射变换矩阵
     */
    void setMatrix(const Matrix& matrix);
    
    /**
     * @brief convert source data to given tensor.
     * @param source    source data.
     * @param iw        source width.
     * @param ih        source height.
     * @param stride    number of elements per row. eg: 100 width RGB contains at least 300 elements.
     * @param dest      given tensor.
     * @return result code.
     * 
     * @brief 将源图像数据转换到目标张量
     * @param source 源图像数据指针
     * @param iw 源图像宽度
     * @param ih 源图像高度
     * @param stride 每行元素数，例如：100宽度的RGB图像至少包含300个元素
     * @param dest 目标张量
     * @return 错误码
     */
    ErrorCode convert(const uint8_t* source, int iw, int ih, int stride, Tensor* dest);
    
    /**
     * @brief convert source data to given tensor.
     * @param source    source data.
     * @param iw        source width.
     * @param ih        source height.
     * @param stride    number of elements per row. eg: 100 width RGB contains at least 300 elements.
     * @param dest      dest data.
     * @param ow      output width.
     * @param oh      output height.
     * @param outputBpp      output bpp, if 0, set as the save and config.destFormat.
     * @param outputStride  output stride, if 0, set as ow * outputBpp.
     * @param type  Only support halide_type_of<uint8_t> and halide_type_of<float>.
     * @return result code.
     * 
     * @brief 将源图像数据转换到目标内存
     * @param source 源图像数据指针
     * @param iw 源图像宽度
     * @param ih 源图像高度
     * @param stride 每行元素数
     * @param dest 目标内存指针
     * @param ow 输出宽度
     * @param oh 输出高度
     * @param outputBpp 输出每像素字节数，0表示使用配置中的destFormat
     * @param outputStride 输出每行字节数，0表示设置为ow * outputBpp
     * @param type 数据类型，仅支持uint8_t和float
     * @return 错误码
     */
    ErrorCode convert(const uint8_t* source, int iw, int ih, int stride, void* dest, int ow, int oh, int outputBpp = 0,
                      int outputStride = 0, halide_type_t type = halide_type_of<float>());
    
    /**
     * @brief create tensor with given data.
     * @param w     image width.
     * @param h     image height.
     * @param bpp   bytes per pixel.
     * @param p     pixel data pointer.
     * @return created tensor.
     * 
     * @brief 创建图像张量（模板版本）
     * @tparam T 数据类型
     * @param w 图像宽度
     * @param h 图像高度
     * @param bpp 每像素字节数
     * @param p 像素数据指针，可选
     * @return 创建的张量指针
     */
    template <typename T>
    static Tensor* createImageTensor(int w, int h, int bpp, void* p = nullptr) {
        return createImageTensor(halide_type_of<T>(), w, h, bpp, p);
    }
    
    /**
     * @brief 创建图像张量
     * @param type 数据类型
     * @param w 图像宽度
     * @param h 图像高度
     * @param bpp 每像素字节数
     * @param p 像素数据指针，可选
     * @return 创建的张量指针
     */
    static Tensor* createImageTensor(halide_type_t type, int w, int h, int bpp, void* p = nullptr);
    
    /**
     * @brief set padding value when wrap=ZERO.
     * @param value     padding value.
     * @return void.
     * 
     * @brief 设置边缘填充值（当wrap=ZERO时）
     * @param value 填充值
     */
    void setPadding(uint8_t value) {
        mPaddingValue = value;
    }
    
    /**
     * @brief set to draw mode.
     * @param void
     * @return void.
     * 
     * @brief 设置为绘制模式
     */
    void setDraw();
    
    /**
     * @brief draw color to regions of img.
     * @param img  the image to draw.
     * @param w  the image's width.
     * @param h  the image's height.
     * @param c  the image's channel.
     * @param regions  the regions to draw, size is [num * 3] contain num x { y, xl, xr }
     * @param num  regions num
     * @param color  the color to draw.
     * @return void.
     * 
     * @brief 在图像的指定区域绘制颜色
     * @param img 要绘制的图像
     * @param w 图像宽度
     * @param h 图像高度
     * @param c 图像通道数
     * @param regions 要绘制的区域，大小为[num * 3]，包含num个{ y, xl, xr }
     * @param num 区域数量
     * @param color 要绘制的颜色
     */
    void draw(uint8_t* img, int w, int h, int c, const int* regions, int num, const uint8_t* color);
private:
    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    ImageProcess(const Config& config);
    
    Matrix mTransform;          ///< 仿射变换矩阵
    Matrix mTransformInvert;    ///< 仿射变换的逆矩阵
    Inside* mInside;            ///< 内部实现指针
    uint8_t mPaddingValue = 0;  ///< 边缘填充值
};
} // namespace CV
} // namespace MNN

#endif /* MNN_ImageProcess_hpp */
