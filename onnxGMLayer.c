#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"

// GameMaker only support String/Double
const wchar_t* char_to_wide_char(const char* str);
__declspec(dllexport) double __stdcall gmortInferenceDouble2Double(const char *model_path, double inputData)
{
    OrtErrorCode ortErrorCode;
    OrtStatus *ortStatus;

    const OrtApiBase *ortApiBase = OrtGetApiBase();
    printf("%s\n", ortApiBase->GetVersionString());
    const OrtApi *ortApi = ortApiBase->GetApi(17);

    OrtEnv *ortEnv = NULL;
    ortApi->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "GMLayerLog", &ortEnv);

    OrtSessionOptions *ortSessionOptions = NULL;
    ortApi->CreateSessionOptions(&ortSessionOptions);

    printf("Model Path: %s\n", model_path);
    printf("wide Model Path: %ls\n", char_to_wide_char(model_path));

    OrtSession *ortSession = NULL;
    ortStatus = ortApi->CreateSession(ortEnv, char_to_wide_char(model_path), ortSessionOptions, &ortSession);
    if (ortStatus != NULL)
        printf("%s\n", ortApi->GetErrorMessage(ortStatus));

    OrtMemoryInfo *ortMemoryInfo = NULL;
    ortApi->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &ortMemoryInfo);

    int64_t inputDims[] = {1};

    OrtValue *inputTensor = NULL;
    OrtValue *outputTensor = NULL;
    ortApi->CreateTensorWithDataAsOrtValue(ortMemoryInfo, &(double)inputData, sizeof((double)inputData),
                                           inputDims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                                           &inputTensor);

    OrtAllocator *ortAllocator = NULL;
    ortApi->GetAllocatorWithDefaultOptions(&ortAllocator);

    const char *inputName;
    ortApi->SessionGetInputName(ortSession, 0, ortAllocator, &inputName);

    const char *outputName;
    ortApi->SessionGetOutputName(ortSession, 0, ortAllocator, &outputName);

    // 执行推理
    ortStatus = ortApi->Run(ortSession, NULL, &inputName, &inputTensor, 1, &outputName, 1, &outputTensor);

    // 获取输出结果
    double *Data;
    ortApi->GetTensorMutableData(outputTensor, (void **)&Data);
    printf("Output: %f\n", Data[0]);

    // clean up
    ortApi->ReleaseSession(ortSession);
    ortApi->ReleaseSessionOptions(ortSessionOptions);
    ortApi->ReleaseEnv(ortEnv);
    ortApi->ReleaseMemoryInfo(ortMemoryInfo);

    return Data[0];
}
__declspec(dllexport) const char *__stdcall gmortGetVersionString()
{
    const OrtApiBase *ortApiBase = OrtGetApiBase();
    return ortApiBase->GetVersionString();
}

// 将 char* 转换为 const wchar_t*
const wchar_t* char_to_wide_char(const char* str) {
    
    // 计算所需的宽字符缓冲区大小
    int length = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);

    // 分配内存并转换
    wchar_t* wstr = (wchar_t*)malloc(length * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, str, -1, wstr, length);

    return wstr; // 返回 const wchar_t*
}

void main()
{
    // OrtErrorCode ortErrorCode;
    // OrtStatus *ortStatus;

    // const OrtApiBase *ortApiBase = OrtGetApiBase();
    // printf("%s\n", ortApiBase->GetVersionString());
    // const OrtApi *ortApi = ortApiBase->GetApi(17);

    // OrtEnv *ortEnv = NULL;
    // ortApi->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "GMLayerLog", &ortEnv);

    // OrtSessionOptions *ortSessionOptions = NULL;
    // ortApi->CreateSessionOptions(&ortSessionOptions);

    // OrtSession *ortSession = NULL;
    // ortStatus = ortApi->CreateSession(ortEnv, L"mlp.onnx", ortSessionOptions, &ortSession);
    // if (ortStatus != NULL)
    //     printf("%s\n", ortApi->GetErrorMessage(ortStatus));

    // OrtMemoryInfo *ortMemoryInfo = NULL;
    // ortApi->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &ortMemoryInfo);

    // float inputData[] = {0.1f}; // 根据模型输入调整
    // int64_t inputDims[] = {1};

    // OrtValue *inputTensor = NULL;
    // OrtValue *outputTensor = NULL;
    // ortApi->CreateTensorWithDataAsOrtValue(ortMemoryInfo, inputData, sizeof(inputData),
    //                                        inputDims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    //                                        &inputTensor);

    // OrtAllocator *ortAllocator = NULL;
    // ortApi->GetAllocatorWithDefaultOptions(&ortAllocator);

    // const char *inputName;
    // ortApi->SessionGetInputName(ortSession, 0, ortAllocator, &inputName);

    // const char *outputName;
    // ortApi->SessionGetOutputName(ortSession, 0, ortAllocator, &outputName);

    // // 执行推理
    // ortStatus = ortApi->Run(ortSession, NULL, &inputName, &inputTensor, 1, &outputName, 1, &outputTensor);

    // // 获取输出结果
    // float *floatData;
    // ortApi->GetTensorMutableData(outputTensor, (void **)&floatData);
    // printf("Output: %f\n", floatData[0]);

    // // clean up
    // ortApi->ReleaseSession(ortSession);
    // ortApi->ReleaseSessionOptions(ortSessionOptions);
    // ortApi->ReleaseEnv(ortEnv);
    // ortApi->ReleaseMemoryInfo(ortMemoryInfo);

    gmortInferenceDouble2Double("C:/workspace/github/GM-OnnxRuntime/mlp.onnx", 0.5);
}