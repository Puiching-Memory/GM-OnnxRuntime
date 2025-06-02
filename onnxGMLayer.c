#include <stdio.h>
#include "onnxruntime_c_api.h"

// GameMaker only support String/Double


onnx_forward(){
}

__declspec(dllexport) const char* GetVersionString(){
    const OrtApiBase* ortApiBase = OrtGetApiBase();
    return ortApiBase->GetVersionString();
}

void main(){
    const OrtApiBase* ortApiBase = OrtGetApiBase();
    printf("%s",ortApiBase->GetVersionString());
    const OrtApi* ortApi = ortApiBase->GetApi(17);

    OrtEnv* ortEnv = NULL;
    ortApi->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE,"GMLayerLog",&ortEnv);

    OrtSession* session = NULL;
    //ortApi->CreateSession(ortEnv,"",);
}