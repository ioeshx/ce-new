# CLIP / FID 评价

此目录保存正式运行的指标输出。计算代码位于 ../metrics.py，完整实验集及采样配置校验位于 ../evaluation.py，统一入口仍为 ../main.py。

## 一次运行采样与指标

先安装 ../requirements.txt，再设置已经完成擦除的检查点：

```bash
cd /home/shx/code/ce-new/casu-2
python -m pip install -r requirements.txt

CHECKPOINT=/path/to/erasure-run/checkpoint \
bash script/sample_and_metrics.sh
```

任务、基础模型、目标概念、FLUX 变体从检查点读取。分辨率、步数、guidance 默认沿用检查点的生成配置；旧检查点缺少配置时回退到该架构默认值。SD1.x 默认 DPM solver、512×512、20 步、CFG 7.5；FLUX.2-klein 沿用原生 4 步、guidance 1、CUDA CPU offload，不强制套用 SD1.x 参数。

默认使用原论文脚本的完整概念集与 data/prompt_template.py 模板：

| task | 测试概念 | 模板/概念 | 图片/提示 | 图片/模式 |
| --- | --- | --- | --- | --- |
| instance | Hello Kitty、Mickey、Pikachu、Snoopy、Spongebob | 80 | 10 | 4000 |
| style | Van Gogh、Monet、Picasso | 30 | 10 | 900 |

每个检查点的评价目录包含 original 与 edit 两个集合，采用相同 prompt、seed 和初始噪声；因此完整 instance 共 8000 张，style 共 1800 张。original 默认共用各架构自己的公共原图目录；配置相同时复用，不同时覆盖该模型、概念的原图。每个样本使用 seed + sample_index，不受尾批次长度影响。沿用旧程序“各模板共用一组样本噪声”的设计，但不声称复现旧程序依赖 batch size 的逐像素随机轨迹。

这是逐检查点评价；原论文的一/二/三目标 instance 设置、三位画家的独立擦除设置，分别传入各自检查点运行即可。不会在此脚本中重新擦除模型。Mickey Mouse/Mickey、SpongeBob/Spongebob、Claude Monet/Monet、Pablo Picasso/Picasso 等别名会用于目标分组，但不会改写生成提示。

```bash
# 自定义输出；完整默认设置仍为全部模板 × 10 张。
CHECKPOINT=/path/to/run/checkpoint \
IMAGE_DIR=/path/to/evaluation-images METRIC_DIR=/path/to/ce-new/casu-2/metric/my-run \
bash script/sample_and_metrics.sh

# 显存较小时，采样批次保持 1，指标批次可单独减小。
CHECKPOINT=/path/to/flux-klein-run/checkpoint \
CPU_OFFLOAD=1 BATCH_SIZE=1 METRIC_BATCH_SIZE=8 \
bash script/sample_and_metrics.sh

# 完全使用已有 CSV 的每条提示，不改写大小写或措辞。
# instance: data/ip_imagenet_prompts.csv；style: data/style_painting_prompts.csv。
CHECKPOINT=/path/to/instance-run/checkpoint \
PROMPT_CSV=/home/shx/code/ce-new/casu-2/data/ip_imagenet_prompts.csv \
bash script/sample_and_metrics.sh

# 仅预览完整生成计划，不加载模型或计算指标。
CHECKPOINT=/path/to/run/checkpoint DRY_RUN=1 \
bash script/sample_and_metrics.sh
```

图片默认位于 ../runs/eval-<擦除运行目录名>/；指标默认位于本目录的 <擦除运行目录名>/。运行目录名可用 RUN_ID 指定。以下文件写入指标目录：

```text
results.json          # 定义、模型版本、逐概念分数与 target/non_target 汇总
per_concept.csv       # concept, role, n, clip_original, clip_edit, clip_delta, fid
per_image_clip.csv    # 每张图的真实提示、seed、CLIP 分数与相对图片路径
cache/                # 按概念/模式保存标准 Inception 特征
```

脚本将采样、指标作为两个独立进程运行，计算指标时生成模型已经退出。支持 DEVICE、DTYPE、CPU_OFFLOAD、HEIGHT、WIDTH、STEPS、GUIDANCE_SCALE、NUM_SAMPLES、SAMPLE_TEMPLATES、CONTENTS、CLIP_MODEL、METRIC_DEVICE、METRIC_BATCH_SIZE、METRIC_WORKERS、LOCAL_FILES_ONLY。额外命令行参数传给采样阶段。

## 指标定义与原实现核对

已参考 /home/shx/code/ce/util/clip_score_cal.py。原实现的 Transformers CLIP 和 torch-fidelity 都可用于这些指标，未发现需要因其库本身而更换实现的理由。新代码继续使用这两个库，torch-fidelity 固定为 0.4.0。

原程序 CLI 的默认 CLIP 模型是 **openai/clip-vit-base-patch16**，虽然后面的 CLIP_Score 类自身默认 large-patch14，但实际运行会被 --version 覆盖。这里采用 CLI 默认值，允许通过 CLIP_MODEL / --clip-model 切换。CLIP 分数为每张图片与其**完整实际生成提示**的归一化特征余弦相似度乘 100，然后逐图片求均值。与旧代码一致，不乘模型的 learned logit_scale、不计算 batch 内交叉配对、不将负值裁剪到 0；因此它与裁剪为非负的 TorchMetrics CLIPScore 定义有区别。

旧实现有以下需要修正的工程问题：

- 从文件名反推提示：标点、路径/空格转义、长文件名等会改变文本。新实现读取 manifest 中保存的原始 prompt。
- 将每个 batch 的均值再平均：最后一个 batch 不满时会被过度加权，且 os.listdir 的顺序会影响这个误差。新实现按图片总数加权，保存逐图分数。旧实验每概念 800/300 张、batch=10 时恰好整除，这个误差通常不会触发。
- 模型未显式 eval，也未关闭 autograd；可能增加内存和计算开销。新实现显式 eval + inference_mode，不宣称旧 CLIP 必然存在 dropout 随机误差。
- 内部张量和 FID 的 cuda=True 硬编码，忽略传入 device。新实现支持 CPU 和指定 CUDA 设备。
- 固定长度截断和未校验的 original 目录可能掩盖文本截断、漏图或错误基准。新实现检查文本长度、所有配对文件、重复路径，以及新数据集记录的检查点 hash。
- 使用追加文本记录作为跳过条件，配置变化可能复用过期分数。新实现每次写结构化结果，FID 特征缓存按文件路径、大小、mtime、特征网络和库版本校验。

FID 保持 torch-fidelity 的 **inception-v3-compat / 2048** 特征、RGB uint8 输入与 TensorFlow 兼容缩放。每个概念单独比较 original 与 edit；不使用真实数据集、不将不同基础模型的 original 混用，不将 compare 拼图纳入统计。目标和非目标分别汇总，fid_macro 表示逐概念 FID 的算术平均，明确区别于将图片混在一起计算的 FID。

FID_METHOD=auto 在样本数小于 2048 时用等价的低秩计算，仍使用完整 2048 维特征，不降维。对中心化特征行矩阵 A、B，协方差平方根项的迹等于 ||A Bᵀ||_* / sqrt((N-1)(M-1))，可避免为每个 300/800 张图的集合求 2048×2048 矩阵的特征值。FID_METHOD=covariance 使用 torch-fidelity 原生统计量计算；两者的数学定义相同，有限精度可能有微小差别。

CLIP target 下降、non_target 保持接近原值，是相应语义变化的证据。非目标 FID 越低，表示与原模型生成分布越接近；目标 FID 是偏离原模型的度量，不能把任意更高数值都当成更好擦除。仍需结合已经检查过的生成图片理解结果。

## 单独采样、复算与断点续跑

```bash
python main.py sample --checkpoint /path/to/run/checkpoint \
  --benchmark --num-samples 10 --sample-template-count 0 \
  --output-dir runs/full-evaluation --no-save-comparisons

python main.py metrics --image-dir runs/full-evaluation \
  --checkpoint /path/to/run/checkpoint --output-dir metric/my-run

# 只重新计算指标；使用之前的 IMAGE_DIR 或默认目录。
CHECKPOINT=/path/to/run/checkpoint SKIP_SAMPLE=1 \
bash script/sample_and_metrics.sh

# 指标放 CPU；采样仍在 CUDA。
CHECKPOINT=/path/to/run/checkpoint METRIC_DEVICE=cpu \
bash script/sample_and_metrics.sh
```

一键脚本默认 RESUME=1，匹配 checkpoint、提示清单和采样配置时跳过已存在且尺寸正确的有效 PNG；漏图/损坏图会重新生成。更换模型、seed、分辨率、步数或提示后，必须使用新 IMAGE_DIR，避免混合实验。断点流程不依赖最终 manifest 已写完。普通 main.py sample 需显式 --resume。大规模采样默认不保存额外 compare 拼图，可通过 --save-comparisons 开启。

original 自动按检查点架构分目录：SD1.x 为 ../runs/original/，SDXL 为 ../runs/original_SDXL/，SD3.0 为 ../runs/original_SD3/，FLUX 为 ../runs/original_FLUX/，不同架构不会互相覆盖。每个根目录内按 <任务>/<概念>/ 保存，例如 instance/snoopy/ 或 style/van_gogh/。图片名为 000000.png、000001.png 等；同目录 manifest.json 逐条记录 filename、prompt、seed、模板及样本编号。config.json 直接保存可读、带缩进的采样参数。直接比较已有 config 和提示清单；一致时复用有效 PNG，不一致时删除该模型、概念的旧 PNG 并重新生成。没有参数 hash、模型文件扫描或文件锁。

实验 manifest 的 original 字段直接保存公共原图的绝对路径；实验内不复制原图，不创建硬链接或软链接。CLIP 使用该条记录的完整 prompt 与原图路径；FID 使用同一条记录的 original/edit 路径。指标加载器允许 original 指向公共目录，并按该目录的 config/manifest 校验概念、prompt、seed 和采样参数；edit 仍从实验目录读取。公共原图被其他采样参数覆盖后，旧实验计算指标会拒绝不匹配的原图。

对已有实验重新运行采样脚本，在公共原图和 edit 均有效时不会再次生成图片；脚本更新 manifest 和网页后，删除该实验本次清单对应的旧 original 文件及空目录。公共原图和 edit 不受删除操作影响。ORIGINAL_DIR / CLI --original-dir 可修改原图根目录，ORIGINAL_CACHE=0 或 --no-original-cache 可关闭公共原图目录；旧目录选项 ORIGINAL_CACHE_DIR / --original-cache-dir 仍可使用。

更改擦除参数时仍使用新的 PAPER_RUN_ROOT/IMAGE_DIR，edit 必须对应新的检查点。双卡任务在 shell 中手动分配，当前 instance 与 style 分别在两张卡上运行，调用方避免同时生成相同任务/概念。旧 hash 目录不再读取；新目录第一次运行会重新生成一次 original。

LOCAL_FILES_ONLY=1 对生成模型与 CLIP 都启用本地加载，并要求 torch-fidelity 的权重已缓存于 $TORCH_HOME/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth（未设置 TORCH_HOME 时使用 PyTorch 默认缓存）。该文件需为 torch-fidelity 的对应权重。

实现依据：[torch-fidelity](https://torch-fidelity.readthedocs.io/en/latest/)、[Transformers CLIP](https://huggingface.co/docs/transformers/v4.56.1/en/model_doc/clip)。

验证：31 项单元检查通过，包括跨检查点原图复用、配置改变后覆盖、图片序号对应、任务隔离和损坏补图，并验证直接读取公共原图、清理旧链接及拒绝不匹配的原图配置；另使用缩小的原生 Klein 模型和缓存的真实 ViT-B/16、Inception 权重，跑通了独立进程的一键采样＋指标流程，并核对新 CLIP 分数与旧归一化余弦公式一致。这些只验证实现，不代表已完成正式完整数据集的擦除实验。

## 原论文六组批量实验

使用 ../script/paper_erase_all.sh 独立擦除 Snoopy、Snoopy+Mickey、Snoopy+Mickey+Spongebob，以及分别擦除 Van Gogh、Monet、Picasso；之后运行 ../script/paper_sample_all.sh。也可直接运行 ../script/paper_all.sh all。默认模型为 SD1.4，每组默认执行完整采样＋CLIP/FID，结果分别位于 metric/paper-sd1/<组名>/。各组单独入口、模型切换、anchor 和输出目录设置见主 README 的第 9 节。

WITH_METRICS=0 bash script/paper_sample_all.sh 仅生成图片。通用 script/sample_and_metrics.sh 也支持 SKIP_METRICS=1；SKIP_SAMPLE=1 则保留仅复算指标的行为。

GPU 直接由 shell 脚本中的 CUDA_VISIBLE_DEVICES 指定。批量生成入口 paper_sample_all.sh 默认单卡；启动时设置 CUDA_VISIBLE_DEVICES=1 可改用 GPU 1。NUM_GPUS=2 启用脚本内写明的两个任务组：GPU 0 依次生成 Snoopy、Snoopy+Mickey+Spongebob、Snoopy+Mickey；GPU 1 依次生成 Van Gogh、Monet、Picasso。可直接修改脚本内的显卡编号和实验命令。每张卡同时最多一组，该组的生成和指标共用这张卡；METRIC_DEVICE=cpu 可单独把指标放到 CPU。擦除始终单卡逐组执行，paper_all.sh all 只在生成阶段使用双卡。单/双卡之间可继续复用已完成图片。完整说明见主 README 第 10 节。
