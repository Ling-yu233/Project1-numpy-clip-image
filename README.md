# 小项目一：用 NumPy 实现 CLIP 图像编码器（ViT-B/32）

本项目只使用 **NumPy**，复现了 OpenCLIP ViT-B-32 图像编码器的前向计算：输入一批图片，
每张图片输出一个 512 维的单位长度特征向量（output token）。模型参数由课程提供，本项目不涉及训练。
数值结果与课程提供的 PyTorch FP32 参考输出逐阶段对比，全部在容差范围内。

完成后的 `image_encoder.py` 会作为期末项目 `szu_python_openclip` 的图像塔继续使用。

## 1. 任务与输入输出

| 项目 | 形状 | 说明 |
|---|---|---|
| 输入 | `[B, 3, 224, 224]` float32 | 经 `preprocess_image.py` 缩放、中心裁剪、归一化后的图片 |
| 内部 token | `[B, 50, 768]` | 1 个 CLS token + 49 个图块 token，经过 12 层 Transformer |
| 输出 | `[B, 512]` float32 | 每张图片一个单位长度的特征向量 |

## 2. 计算流程

```
图片 [B,3,224,224]
  │ patchify（TODO-01）     切成 7×7 个 32×32 图块，每块按 C,H,W 展平
  ▼ [B,49,3072]
  │ embed（TODO-05）        图块线性投影（无 bias）→ 前面拼接 CLS → 加位置编码
  ▼ [B,50,768]
  │ ln_pre
  │ 12 × TransformerBlock（TODO-02/03/04）
  │    x = x + MultiHeadAttention(LN1(x))        12 个头，每头 64 维
  │    x = x + MLP(LN2(x))                       768 → 3072 → GELU → 768
  ▼ [B,50,768]
  │ 输出（TODO-06）         取第 0 行 CLS → ln_post → 乘 visual.proj → 单位化
  ▼ [B,512]
```

## 3. 我完成的部分（均在 `image_encoder.py` 中）

| 编号 | 位置 | 实现内容 |
|---|---|---|
| TODO-01 | `ImageEncoder.patchify` | `reshape(B,3,7,32,7,32)` → `transpose(0,2,4,1,3,5)` → `reshape(B,49,3072)`，并检查输入形状 |
| TODO-02 | `MultiHeadAttention.split_heads` | `[B,N,768]` → `[B,N,12,64]` → `transpose` 成 `[B,12,N,64]` |
| TODO-03 | `MultiHeadAttention.__call__` | `scores = q @ kᵀ / √64`（只交换 k 的最后两个轴）；`context = coefficients @ v`；合并多头；输出投影 |
| TODO-04 | `TransformerBlock.__call__` | 第一次残差相加；MLP（`fc1` → GELU → `fc2`）；第二次残差相加 |
| TODO-05 | `ImageEncoder.embed` | 图块投影 `patches @ W.T`；沿 token 轴拼接 CLS；广播加位置编码 |
| TODO-06 | `ImageEncoder.__call__` | 取 CLS 行 → post-Norm → 乘 `visual.proj`（已是 `[in,out]`，不转置）→ 单位化 |

其余文件（`helpers.py`、`check.py`、`run.py`、`runtime.py`、`preprocess_image.py` 等）由课程提供，未改动。

## 4. 目录结构

```
project1-numpy-student/
├── image_encoder.py        # ★ 本人实现：图像编码器（TODO-01 ~ TODO-06）
├── helpers.py              # 课程提供：LayerNorm、GELU、softmax、单位化等
├── preprocess_image.py     # 课程提供：图片缩放、裁剪、归一化
├── runtime.py              # 课程提供：加载权重与图片
├── run.py                  # 课程提供：命令行，图片 → 输出向量 .npz
├── check.py                # 课程提供：与参考输出逐阶段比对
├── transformer_steps.py    # 课程提供：微型 Transformer 计算演示
├── config.json             # 模型结构参数
├── requirements.txt        # 依赖清单
├── PPT-CODE-MAP.txt        # 课件与代码行的对应表
├── SOURCE-MANIFEST.json    # 文件来源与校验值
├── weights/
│   ├── manifest.json       # 权重张量名、形状、SHA-256
│   └── image_encoder.npz   # （未上传，见第 5 节）
├── assets/
│   ├── images.json         # 10 张测试图片的 ID 与来源
│   ├── manifest.json
│   ├── reference_outputs.npz  # 参考输出，仅供 check.py 比对
│   └── images/             # 10 张测试图片 image_01.jpg ~ image_10.jpg
└── lecture-demos/          # 课堂 NumPy 示例脚本
```

## 5. 数据与权重获取

测试图片已包含在仓库中。以下文件**没有上传**，需要从课程平台下载的附件中取得，放回对应位置：

| 文件 | 原因 | 放置位置 |
|---|---|---|
| `image_encoder.npz`（312 MB） | 超过 GitHub 单文件 100 MB 上限 | `weights/` |
| `project1_exercises.pdf` | 课程练习材料，与代码运行无关 | `practice/` |

放好后运行 `python check.py --stage assets`。程序会用 `weights/manifest.json` 中的 SHA-256 校验文件是否正确。

## 6. 环境与运行

**环境**：Python 3.9 及以上（本人在 Windows 11、Python 3.12 下测试），不需要 GPU、PyTorch 或网络。

```bash
pip install -r requirements.txt      # numpy、Pillow、imageio
```

**数值检查**（在本目录下运行）：

```bash
python check.py --stage all          # 运行全部阶段
python check.py --stage block0       # 只检查某一阶段
```

可选阶段：`assets`、`preprocess`、`patches`、`tokens`、`block0`、`last_tokens`、`output`、`batch`。

**提取图片特征**：

```bash
python run.py --output output_tokens.npz                              # 10 张自带图片
python run.py --images a.jpg b.png --output my_tokens.npz             # 自己的图片
```

输出的 `.npz` 文件包含 `ids`（图片名）和 `output_tokens`（`[B,512]`）。
在本人电脑上，10 张图片的完整检查约需 10 秒（`helpers.py` 提供的 GELU 用 Python 循环逐个计算 erf，是主要耗时）。

## 7. 运行结果

`python check.py --stage all` 的输出（容差：最终输出 atol=2e-5，内部 token atol=2e-4，rtol 均为 2e-4）：

| 阶段 | 形状 | 最大绝对误差 | 平均绝对误差 | 结果 |
|---|---|---|---|---|
| preprocess | (10, 3, 224, 224) | 0 | 0 | PASS |
| patches | (2, 49, 3072) | 0 | 0 | PASS |
| tokens | (2, 50, 768) | 4.578e-05 | 1.321e-06 | PASS |
| block0 | (2, 50, 768) | 6.109e-06 | 3.322e-07 | PASS |
| last_tokens | (2, 50, 768) | 4.101e-05 | 2.748e-06 | PASS |
| output | (10, 512) | 5.960e-07 | 8.427e-08 | PASS |
| batch | (3, 512) | 4.657e-07 | 7.892e-08 | PASS |

`patches` 只做重排，误差为 0；其余阶段的误差来自浮点矩阵乘法的舍入，经过 12 层后没有累积放大。
`batch` 阶段用顺序打乱并含重复的图片（`[1,0,1]`）测试，说明实现不依赖图片在批次中的位置。

## 8. 实现要点

1. **线性层权重存储为 `[out, in]`**：注意力和 MLP 的权重都要写 `x @ W.T`；只有 `visual.proj` 已经是 `[in, out]`，直接 `x @ proj`。
2. **多头拆分必须先 reshape 再 transpose**：直接 `reshape(B,12,N,64)` 形状虽然正确，但数据会错位，程序也不会报错。
3. **4D 的 k 只交换最后两个轴**：用 `k.transpose(0,1,3,2)`，不能用 `k.T`（`k.T` 会反转全部四个轴）。
4. **保持 float32**：缩放因子写 `np.sqrt(np.float32(64))`，避免 NumPy 2 把结果提升为 float64，导致检查失败。
5. **`causal` 参数保留**：图像编码器不使用因果掩码，但期末项目的文本编码器要用它。

## 9. 参考与说明

- 模型结构：OpenCLIP ViT-B-32（Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021）。
- 课程提供的代码框架、权重、测试图片与参考输出，来源详见 `SOURCE-MANIFEST.json` 与 `assets/images.json`。
- 使用 AI 辅助的情况已在实验报告中说明。
