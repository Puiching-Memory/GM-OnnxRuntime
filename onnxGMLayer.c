#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"

// GameMaker only support String/Double

const wchar_t *char_to_wide_char(const char *str);

static OrtEnv *ortEnv = NULL;
static OrtSessionOptions *ortSessionOptions = NULL;
static const OrtApi *ortApi = NULL;
OrtSession *ortSession = NULL;
OrtAllocator *ortAllocator = NULL;
const char *inputName = NULL;
const char *outputName = NULL;

OrtErrorCode ortErrorCode;
OrtStatus *ortStatus;

__declspec(dllexport) void __stdcall ortInit()
{
    if (ortEnv == NULL)
    {
        const OrtApiBase *ortApiBase = OrtGetApiBase();
        ortApi = ortApiBase->GetApi(17);
        ortApi->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "GMLayerLog", &ortEnv);
        ortApi->CreateSessionOptions(&ortSessionOptions);
    }
}

__declspec(dllexport) void __stdcall ortFree()
{
    if (ortEnv)
    {
        ortApi->ReleaseSessionOptions(ortSessionOptions);
        ortApi->ReleaseEnv(ortEnv);
        ortApi->ReleaseSession(ortSession);
    }
}

__declspec(dllexport) void __stdcall ortLoadModelFromFile(const char *model_path)
{
    printf("Model Path: %s\n", model_path);
    printf("wide Model Path: %ls\n", char_to_wide_char(model_path));

    ortStatus = ortApi->CreateSession(ortEnv, char_to_wide_char(model_path), ortSessionOptions, &ortSession);
    if (ortStatus != NULL)
        printf("CreateSession:%s\n", ortApi->GetErrorMessage(ortStatus));

    ortApi->GetAllocatorWithDefaultOptions(&ortAllocator);
    ortApi->SessionGetInputName(ortSession, 0, ortAllocator, &inputName);
    ortApi->SessionGetOutputName(ortSession, 0, ortAllocator, &outputName);
}

__declspec(dllexport) double __stdcall ortRunDouble(double *inputData, uint64_t inputDataLength, uint64_t *inputShape)
{
    OrtValue *inputTensor = NULL;
    OrtValue *outputTensor = NULL;
    static OrtMemoryInfo *ortMemoryInfo = NULL;
    ortApi->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &ortMemoryInfo);

    ortStatus = ortApi->CreateTensorWithDataAsOrtValue(ortMemoryInfo,
                                                       inputData,
                                                       sizeof(double) * inputDataLength,
                                                       inputShape,
                                                       sizeof(inputShape) / sizeof(inputShape[0]),
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                                                       &inputTensor);
    if (ortStatus != NULL)
        printf("CreateTensorWithDataAsOrtValue:%s\n", ortApi->GetErrorMessage(ortStatus));

    // 执行推理
    ortStatus = ortApi->Run(ortSession, NULL, &inputName, &inputTensor, 1, &outputName, 1, &outputTensor);
    if (ortStatus != NULL)
        printf("Run:%s\n", ortApi->GetErrorMessage(ortStatus));

    // 获取输出结果
    double *Data;
    ortApi->GetTensorMutableData(outputTensor, (void **)&Data);
    if (ortStatus != NULL)
        printf("GetTensorMutableData:%s\n", ortApi->GetErrorMessage(ortStatus));
    
    
    // 获取输出张量的形状
    OrtTensorTypeAndShapeInfo *tensorInfo = NULL;
    ortApi->GetTensorTypeAndShape(outputTensor, &tensorInfo);

    size_t outputShape;
    ortApi->GetTensorShapeElementCount(tensorInfo, &outputShape);

    for (int i = 0; i < outputShape; i++)
    {
        printf("%f,", Data[i]);
        inputData[i] = Data[i]; // 将结果存储回输入数据
    }

    // 释放资源
    ortApi->ReleaseMemoryInfo(ortMemoryInfo);
    ortApi->ReleaseValue(inputTensor);
    ortApi->ReleaseValue(outputTensor);

    return outputShape;
}

__declspec(dllexport) const char *__stdcall ortGetVersionString()
{
    const OrtApiBase *ortApiBase = OrtGetApiBase();
    return ortApiBase->GetVersionString();
}

// 将 char* 转换为 const wchar_t*
const wchar_t *char_to_wide_char(const char *str)
{

    // 计算所需的宽字符缓冲区大小
    int length = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);

    // 分配内存并转换
    wchar_t *wstr = (wchar_t *)malloc(length * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, str, -1, wstr, length);

    return wstr; // 返回 const wchar_t*
}

void main()
{
    ortInit();
    printf("ONNX Runtime Version: %s\n", ortGetVersionString());
    ortLoadModelFromFile("C:/workspace/github/GM-OnnxRuntime/mlp.onnx");
    double inputData[] = {0.1, 0.2, 0.3};
    uint64_t inputShape[] = {3};
    ortRunDouble(inputData, sizeof(inputData) / sizeof(inputData[0]) , inputShape);
    ortFree();
}