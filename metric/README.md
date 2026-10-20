# CLIP / FID 评价

本目录保存指标输出。计算代码在 `../metrics.py`，评价集与续跑校验在 `../evaluation.py`。统一入口为 `main.py metrics`。

## 生成图片并计算指标

```bash
cd /home/shx/code/ce-new/casu-2
bash script/sample.sh instance_snoopy

# 指定已有检查点、图片目录和指标目录。
bash script/sample.sh instance_snoopy \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/images --metric-dir /path/to/metrics
```

参数直接修改 `script/sample.sh` 开头的变量和数组，不需要设置实验环境变量。生成与指标分两个 Python 进程执行，计算指标时生成模型已经释放。模型、任务、目标与默认采样配置从检查点读取。

| task | 评价概念 | 模板/概念 | 图片/提示 | original / edit 各自图片数 |
| --- | --- | --- | --- | --- |
| instance | Hello Kitty、Mickey、Pikachu、Snoopy、Spongebob | 80 | 10 | 4000 |
| style | Van Gogh、Monet、Picasso | 30 | 10 | 900 |

默认图片位于 `runs/paper-<模型>/images/<实验名>/`，指标位于 `metric/paper-<模型>/<实验名>/`。`--run-root` 或脚本顶部的 RUN_ROOT 可指定实验根目录。

```text
results.json          # 配置、定义、逐概念结果及目标/非目标汇总
per_concept.csv       # 每个概念的原图/编辑图 CLIP、差值及 FID
per_image_clip.csv    # 每张图的提示、seed 和 CLIP 分数
cache/                # Inception 特征缓存
```

脚本顶部的 WITH_METRICS、SKIP_SAMPLE 控制阶段；METRIC_DEVICE、METRIC_BATCH_SIZE、CLIP_MODEL、FID_METHOD 控制指标。额外的 Python 采样参数放在 SAMPLE_OPTIONS，指标参数放在 METRIC_OPTIONS。命令行也支持：

```bash
bash script/sample.sh instance_snoopy --no-metrics
bash script/sample.sh instance_snoopy --metrics-only
bash script/sample.sh instance_snoopy --metric-device cpu
bash script/sample.sh instance_snoopy --local-files-only
bash script/sample.sh instance_snoopy --prompt-csv data/ip_imagenet_prompts.csv
```

## 指标定义与原实现核对

已参考 /home/shx/code/ce/util/clip_score_cal.py。原实现的 Transformers CLIP 和 torch-fidelity 都可用于这些指标，未发现需要因其库本身而更换实现的理由。新代码继续使用这两个库，torch-fidelity 固定为 0.4.0。

原程序 CLI 的默认 CLIP 模型是 **openai/clip-vit-base-patch16**，虽然后面的 CLIP_Score 类自身默认 large-patch14，但实际运行会被 --version 覆盖。这里采用 CLI 默认值，允许通过 `sample.sh` 顶部的 CLIP_MODEL 或 Python 的 --clip-model 切换。CLIP 分数为每张图片与其**完整实际生成提示**的归一化特征余弦相似度乘 100，然后逐图片求均值。与旧代码一致，不乘模型的 learned logit_scale、不计算 batch 内交叉配对、不将负值裁剪到 0；因此它与裁剪为非负的 TorchMetrics CLIPScore 定义有区别。

旧实现有以下需要修正的工程问题：

- 从文件名反推提示：标点、路径/空格转义、长文件名等会改变文本。新实现读取 manifest 中保存的原始 prompt。
- 将每个 batch 的均值再平均：最后一个 batch 不满时会被过度加权，且 os.listdir 的顺序会影响这个误差。新实现按图片总数加权，保存逐图分数。旧实验每概念 800/300 张、batch=10 时恰好整除，这个误差通常不会触发。
- 模型未显式 eval，也未关闭 autograd；可能增加内存和计算开销。新实现显式 eval + inference_mode，不宣称旧 CLIP 必然存在 dropout 随机误差。
- 内部张量和 FID 的 cuda=True 硬编码，忽略传入 device。新实现支持 CPU 和指定 CUDA 设备。
- 固定长度截断和未校验的 original 目录可能掩盖文本截断、漏图或错误基准。新实现检查文本长度、所有配对文件、重复路径，以及新数据集记录的检查点 hash。
- 使用追加文本记录作为跳过条件，配置变化可能复用过期分数。新实现每次写结构化结果，FID 特征缓存按文件路径、大小、mtime、特征网络和库版本校验。

FID 保持 torch-fidelity 的 **inception-v3-compat / 2048** 特征、RGB uint8 输入与 TensorFlow 兼容缩放。每个概念单独比较 original 与 edit；不使用真实数据集、不将不同基础模型的 original 混用，不将 compare 拼图纳入统计。目标和非目标分别汇总，fid_macro 表示逐概念 FID 的算术平均，明确区别于将图片混在一起计算的 FID。

`sample.sh` 中 FID_METHOD="auto" 在样本数小于 2048 时用等价的低秩计算，仍使用完整 2048 维特征，不降维。对中心化特征行矩阵 A、B，协方差平方根项的迹等于 ||A Bᵀ||_* / sqrt((N-1)(M-1))，可避免为每个 300/800 张图的集合求 2048×2048 矩阵的特征值。FID_METHOD="covariance" 使用 torch-fidelity 原生统计量计算；两者的数学定义相同，有限精度可能有微小差别。

CLIP target 下降、non_target 保持接近原值，是相应语义变化的证据。非目标 FID 越低，表示与原模型生成分布越接近；目标 FID 是偏离原模型的度量，不能把任意更高数值都当成更好擦除。仍需结合已经检查过的生成图片理解结果。

## 单独采样、复算与续跑

```bash
python main.py sample --checkpoint /path/to/checkpoint \
  --benchmark --num-samples 10 --sample-template-count 0 \
  --output-dir runs/full-evaluation --save-comparisons --resume

python main.py metrics --image-dir runs/full-evaluation \
  --checkpoint /path/to/checkpoint --output-dir metric/my-run \
  --metrics clip,fid --device cuda:0
```

`sample.sh` 默认带 `--resume`，跳过配置匹配且有效的 PNG，补齐缺图或损坏图。更改检查点、seed、分辨率或提示时使用新的图片目录。

公共原图按架构保存在 `runs/original/`、`runs/original_SDXL/`、`runs/original_SD3/`、`runs/original_FLUX/`，每个根目录下按任务 / 概念组织。配置和提示相同时复用；否则重新生成该概念的原图。实验目录中的 `original` 是指向公共原图任务目录的相对软链接，manifest 和网页通过该链接引用图片，不复制原图；指标计算仍会核对公共原图的配置。`--original-dir` 可指定其他根目录，`--no-original-cache` 关闭公共原图复用。

`--local-files-only` 同时传给生成和指标阶段。FID 要求对应的 torch-fidelity Inception 权重已缓存在 PyTorch 的 hub/checkpoints 目录。

## 六组批量评价

```bash
bash script/paper_erase_all.sh
bash script/paper_sample_all.sh
```

单卡 / 双卡和设备编号直接修改 `paper_sample_all.sh` 顶部的 NUM_GPUS、DEVICE0、DEVICE1。每组采样结束后计算该组指标，默认使用相同设备；METRIC_DEVICE 可另设为 CPU。详细实验列表和参数说明见 [主 README](../README.md)。
