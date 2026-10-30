# VLA 的 SONIC low_latency 模型

权重从 NVIDIA GEAR-SONIC 的 `low_latency` 发布目录下载；在项目根目录执行 `uv run --no-project --with huggingface_hub python download_from_hf.py --low-latency`。权重与 TensorRT 缓存不随源码提交；缓存由目标机器生成，电脑和 G1 分别生成。

| 文件 | SHA-256 |
|---|---|
| `model_decoder.onnx` | `c4ac2e74045e7cbfb568f15e6bf47ea7ce023df7a94322af50be223e0a628bab` |
| `model_encoder.onnx` | `60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b` |
| `observation_config.yaml` | `582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab` |
