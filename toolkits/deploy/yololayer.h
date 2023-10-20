#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include "NvInfer.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 1;
    static constexpr int INPUT_H = 384;
    static constexpr int INPUT_W = 640;
    static constexpr int IMG_H = 360;
    static constexpr int IMG_W = 640;
    // static constexpr int INPUT_H = 192;
    // static constexpr int INPUT_W = 320;
    // static constexpr int IMG_H = 180;
    // static constexpr int IMG_W = 320;

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2IOExt
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int getNbOutputs() const override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        int initialize() override;

        virtual void terminate() override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() const override;

        virtual void serialize(void* buffer) const override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const override;

        const char* getPluginVersion() const override;

        void destroy() override;

        IPluginV2IOExt* clone() const override;

        void setPluginNamespace(const char* pluginNamespace) override;

        const char* getPluginNamespace() const override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

        void detachFromContext() override;

    private:
        void forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const override;

        const char* getPluginVersion() const override;

        const PluginFieldCollection* getFieldNames() override;

        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

        void setPluginNamespace(const char* libNamespace) override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif
