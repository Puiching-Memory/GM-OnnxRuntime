

# GameMaker-OnnxRuntime

一个插件，为gamemaker提供开箱即用的onnxruntime支持API

# 快速开始

```gml
show_debug_message(gmortInferenceDouble2Double("C:/workspace/github/GM-OnnxRuntime/mlp.onnx",0.5))
show_debug_message(gmortGetVersionString())
```

# TODO

* [ ] 多维数组输入/输出
* [ ] onnx模型内存复用
* [ ] 可配置日志级别
* [ ] linux平台支持
* [ ] GPU支持
* [ ] winML支持

# 文档

Windows自带onnxruntime: [https://learn.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions](https://learn.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions)

通过缓冲区突破GM类型限制，参考：[https://github.com/YAL-GameMaker-Tools/GmlCppExtFuncs](https://github.com/YAL-GameMaker-Tools/GmlCppExtFuncs)
