# DocDiff Inference Script

这是一个独立的推理脚本，用于在Linux服务器上运行DocDiff模型进行文档增强处理。

## 功能特性

- 支持单张图像和批量图像处理
- 自动检测GPU/CPU设备
- 支持多种图像格式 (JPG, PNG, BMP, TIFF等)
- 可选择保存中间结果
- 支持DPM solver快速推理
- 支持原生分辨率处理

## 环境要求

```bash
pip install torch torchvision pillow pyyaml tqdm numpy
```

## 使用方法

### 1. 单张图像推理

```bash
python inference.py --input /path/to/input/image.jpg --output /path/to/output/result.png
```

### 2. 批量图像推理

```bash
python inference.py --input /path/to/input/folder --output /path/to/output/folder
```

### 3. 完整参数示例

```bash
python inference.py \
  --config conf.yml \
  --input /path/to/input/image.jpg \
  --output /path/to/output/result.png \
  --save_intermediate \
  --init_model checksave/init.pth \
  --denoiser_model checksave/denoiser.pth
```

## 参数说明

- `--config`: 配置文件路径 (默认: conf.yml)
- `--input`: 输入图像路径或目录 (必需)
- `--output`: 输出图像路径或目录 (必需)
- `--save_intermediate`: 保存中间结果 (可选)
- `--init_model`: 初始预测器模型路径 (默认: checksave/init.pth)
- `--denoiser_model`: 去噪器模型路径 (默认: checksave/denoiser.pth)

## 配置文件

如果没有配置文件，脚本会使用默认参数。你也可以修改 `conf.yml` 文件来调整模型参数：

```yaml
# 模型参数
IMAGE_SIZE: [128, 128]
CHANNEL_X: 3
CHANNEL_Y: 3
TIMESTEPS: 100
SCHEDULE: 'linear'
MODEL_CHANNELS: 32
NUM_RESBLOCKS: 1
CHANNEL_MULT: [1, 2, 3, 4]

# 推理参数
PRE_ORI: 'True'           # 预测x0而非噪声
DPM_SOLVER: 'False'       # 使用DPM solver
DPM_STEP: 20              # DPM solver步数
NATIVE_RESOLUTION: 'False' # 原生分辨率处理
```

## 模型文件

确保你有以下预训练模型文件：
- `checksave/init.pth`: 初始预测器权重
- `checksave/denoiser.pth`: 去噪器权重

## 输出格式

- 默认模式：只保存最终增强结果
- 中间结果模式 (`--save_intermediate`): 保存拼接图像，包含：
  - 输入图像
  - 初始预测
  - 采样结果
  - 最终结果

## 注意事项

1. 输入图像分辨率必须是8的倍数，脚本会自动调整
2. 推荐使用GPU加速推理
3. 批量处理时，输出文件名会自动添加 `_enhanced.png` 后缀
4. 如果模型文件不存在，脚本会报错并退出

## 故障排除

### 常见错误

1. **模型文件未找到**
   ```
   Error loading pretrained weights: [Errno 2] No such file or directory
   ```
   解决方法：检查模型文件路径是否正确

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决方法：使用较小的批量大小或启用原生分辨率处理

3. **图像格式不支持**
   ```
   Error loading image: cannot identify image file
   ```
   解决方法：确保图像格式为JPG、PNG、BMP或TIFF

### 性能优化

- 使用GPU进行推理
- 对于大图像，启用 `NATIVE_RESOLUTION: 'True'`
- 如果速度要求较高，可以启用 `DPM_SOLVER: 'True'`

## 示例

处理文档去模糊：
```bash
python inference.py --input blur_document.jpg --output enhanced_document.png
```

批量处理文档：
```bash
python inference.py --input ./documents/ --output ./enhanced_documents/
```

保存处理过程：
```bash
python inference.py --input document.jpg --output result.png --save_intermediate
``` 