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

__declspec(dllexport) void __stdcall ortInit(int ortLoggingLevel)
{
    if (ortEnv == NULL)
    {
        const OrtApiBase *ortApiBase = OrtGetApiBase();
        ortApi = ortApiBase->GetApi(17);
        ortApi->CreateEnv(ortLoggingLevel, "GMLayerLog", &ortEnv);
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

__declspec(dllexport) double *__stdcall ortRunDouble(double *inputData, int64_t *inputDims)
{
    OrtValue *inputTensor = NULL;
    OrtValue *outputTensor = NULL;
    static OrtMemoryInfo *ortMemoryInfo = NULL;
    ortApi->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &ortMemoryInfo);

    ortApi->CreateTensorWithDataAsOrtValue(ortMemoryInfo, inputData, inputDims[0] * sizeof(double),
                                           inputDims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                                           &inputTensor);

    // 执行推理
    ortStatus = ortApi->Run(ortSession, NULL, &inputName, &inputTensor, 1, &outputName, 1, &outputTensor);
    if (ortStatus != NULL)
        printf("Run:%s\n", ortApi->GetErrorMessage(ortStatus));

    // 获取输出结果
    double *Data;
    ortApi->GetTensorMutableData(outputTensor, (void **)&Data);
    if (ortStatus != NULL)
        printf("GetTensorMutableData:%s\n", ortApi->GetErrorMessage(ortStatus));

    // 释放资源
    ortApi->ReleaseMemoryInfo(ortMemoryInfo);
    ortApi->ReleaseValue(inputTensor);
    ortApi->ReleaseValue(outputTensor);

    return Data;
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
    ortInit(ORT_LOGGING_LEVEL_VERBOSE);
    printf("ONNX Runtime Version: %s\n", ortGetVersionString());
    ortLoadModelFromFile("C:/workspace/github/GM-OnnxRuntime/mlp.onnx");
    int64_t inputDims[] = {3};
    double inputData[] = {0.1, 0.2, 0.3};
    double *data = ortRunDouble(inputData, inputDims);

    for (int i = 0; i < 3; i++)
    {
        printf("%f,", data[i]);
    }

    ortFree();
}