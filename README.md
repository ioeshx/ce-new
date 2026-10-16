# CASU-2：SD 1.x、SDXL、SD3.0 与 FLUX 概念擦除

独立实现于本目录，不修改、不导入 aaai27/casu_code 的源码。现有 data 数据文件保持原样；data/prompt_template.py 是对原 prompt_templates.py 的兼容入口。

支持**闭式擦除 → 保存编辑检查点 → 生成同噪声的原模型/编辑模型图片**，以及后续的**完整实验集采样 → CLIP / FID 评价**。擦除入口默认仅生成快速对照图；完整指标通过独立命令运行。支持 SD1.x（默认 SD1.4）、SDXL base、SD3.0 medium，以及 FLUX.2-klein-4B；兼容 FLUX.1-dev / schnell。

实现模块直接放在本目录，不再保留额外的 casu 子包和独立入口包装文件：

~~~text
casu-2/
  main.py          # 命令行实现：erase / sample / metrics
  models.py       # 架构选择、加载与检查点格式识别
  sd.py           # SD1.x 适配、编辑控制与检查点
  sdxl.py         # SDXL 的 K/V 与 pooled 文本列
  sd3.py          # SD3.0 的文本序列与 pooled 条件入口
  flux.py         # Klein 的 Qwen3 入口；兼容 FLUX.1 的 T5/pooled CLIP
  conditioning.py # 多编码器实际表示、分段 token 对齐
  prompts.py      # 提示对、token 对齐与保留文本
  solver.py       # 零空间闭式求解
  sampling.py     # 图片采样与对照网页
  originals.py    # 原图按任务/概念保存，直接比较 config 后复用或覆盖
  evaluation.py   # 论文完整实验集、CSV 提示和续跑配置校验
  metrics.py      # CLIP / FID，目标与非目标逐概念评价
  selection.py    # 逐模块删除与判别
  selection_proxy.py # 原生管线轨迹上的模块组合排序
  data/
  metric/         # 指标输出与详细评价说明
  script/
  tests/
~~~

统一使用 python main.py erase、sample 和 metrics；各自追加 --help 查看参数。数据、默认 runs 与 metric 目录均定位到本目录。已有 SD1.x 检查点可以继续加载；新检查点记录自己的架构，采样时自动识别。

## 1. 安装与首次运行

建议使用 Python 3.10+，在独立环境中安装：

~~~bash
cd /home/shx/code/ce-new/casu-2
python -m pip install -r requirements.txt

# 实例：Snoopy → dog，开启 CA，关闭 selective update。
bash script/sd14_instance.sh

# 风格：Van Gogh → realistic，开启 CA，关闭 selective update。
bash script/sd14_style.sh
~~~

FLUX.2-klein 需要新版依赖，本项目已将 Diffusers 从 0.36.0 更新到 0.37.1；已有环境也需重新安装 requirements.txt。代码不会自动修改当前 Python 环境。

默认使用 CompVis/stable-diffusion-v1-4，512×512、20 步、每个提示 1 张。脚本默认生成前 8 个模板的前后对照；设置 SAMPLE_TEMPLATES=0 使用全部模板：instance 为 imagenet_templates，style 为 painting_templates。

可以传入本地完整 Diffusers 模型目录，也可以替换为 SD1.5 模型。单文件 ckpt/safetensors 基础模型暂不支持；本实现产生的编辑检查点则是专用的局部权重格式。

~~~bash
SD_CKPT=/path/to/sd14 \
TARGETS="Mickey Mouse" ANCHORS="mouse" \
SAMPLE_TEMPLATES=0 NUM_SAMPLES=2 \
OUTPUT_DIR=/path/to/new-run \
bash script/sd14_instance.sh --local-files-only
~~~

所有脚本都可以从其他工作目录调用。相对 output/retain 路径相对当前工作目录解析，默认数据路径则固定到本项目 data。

运行目录非空时默认拒绝覆盖。确需复用时显式传入 --overwrite；建议每次使用新的输出目录。

## 2. CA 与 selective update 的开关

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| --contextual-anchor / --no-contextual-anchor | 开启 | 成对上下文拟合 / 独立概念提示拟合 |
| --selective-update / --no-selective-update | 关闭 | 是否启用生成反馈的逐模块删除 |
| --selection-judge manual / vlm | manual | 人工查看图片判定 / 多模态接口判定 |
| --params V / K / KV | V | 编辑的文本条件投影；KV 在同一注意力层中成组撤去 |
| --token-mode full / subject | full | 全文本序列对齐 / 仅发生替换的 token 跨度 |
| --architecture sd1 / sdxl / sd3 / flux | sd1 | 编辑架构；采样时默认从检查点读取 |
| --condition-branches both / tokens / pooled | both | SDXL/SD3/FLUX 的条件通路选择 |
| --max-fit-tokens | 16 | 新架构每个提示、每段序列的拟合行预算；0 使用全部 |
| --max-sequence-length | Klein 512，其他 256 | Qwen3/T5 文本长度 |
| --flux-variant klein / dev / schnell | klein | FLUX.2-klein-4B / FLUX.1-dev / FLUX.1-schnell |
| --encode-batch-size | Klein 1，其他 4 | 文本编码批次 |
| --model-cpu-offload / --no-model-cpu-offload | Klein 在 CUDA 上开启，其他关闭 | 整模型 CPU offload，仍需要 CUDA |

--params 只控制 SD1/SDXL 的 K/V；SD3/FLUX 编辑文本条件入口线性投影，不把其内部共享投影当成 U-Net 的 K/V。

Shell 脚本通过 CA 和 SU 控制同样的开关：

~~~bash
CA=1 SU=0 bash script/sd14_instance.sh
CA=0 SU=0 bash script/sd14_instance.sh
CA=1 SU=1 bash script/sd14_instance.sh
CA=0 SU=1 bash script/sd14_instance.sh
~~~

开启 SU 会明显增加生成次数。manual 模式会在每轮打印图片网页路径，并等待输入：

- no：所有图片均未见任何待擦除目标，接受本次删除。
- yes：至少一张图片重新出现任一目标，恢复本轮撤去的更新并停止。
- unknown：无法确认，停止并保留上一轮模型。

第一次判断是“全部候选更新”的模型。如果它未通过，不尝试删除，保存该模型及未通过的状态供检查。没有输入或接口失败绝不自动视为通过。初始未通过、判断无法确定或所有更新均可撤去等待复查的情况，会在保存结果后以非零退出码结束。

### 直接调用 Python

~~~bash
python main.py erase \
  --task instance \
  --target-concepts "Snoopy" --anchor-concepts "dog" \
  --contextual-anchor --no-selective-update \
  --params V --beta 1.0 \
  --context-template-count 30 \
  --retain-count 128 \
  --sample-template-count 8 --num-samples 1 \
  --contents "Snoopy,Mickey Mouse,Pikachu" \
  --output-dir runs/snoopy-ca

python main.py erase \
  --task style \
  --target-concepts "Van Gogh" --anchor-concepts "realistic" \
  --contextual-anchor --no-selective-update \
  --sample-template-count 8 \
  --output-dir runs/van-gogh-ca
~~~

多个目标用逗号分隔。可以给一个共享 anchor，也可以按相同顺序给出等量 anchors：

~~~bash
TARGETS="Snoopy,Mickey Mouse" ANCHORS="dog,mouse" \
bash script/sd14_instance.sh
~~~

anchor 应由使用者根据概念范围选择。这里的 dog、mouse、realistic 是初始实验参考，不是已经验证安全或最优的 anchor。本版不自动搜索多个 anchor 候选，也不做 LLM 模板均值或 anchor 梯度优化。

关闭 CA 后，使用裸目标名与裸 anchor 构造拟合数据，而不把目标名重复映射到多个上下文均值。空 anchor 可用 --anchor-concepts ""；开启 CA 时只移除模板占位内容，保留其余上下文，部分模板可能语法不自然，建议优先显式指定替代描述。

## 3. 实现的方法

### 编辑入口与实际输入

候选入口为 SD1.x U-Net 中 attn2.to_k / attn2.to_v。默认只编辑 V，每个 attn2 是一个选择单元；设置 KV 时同一层 K/V 一起保留或撤去。图像 Q、自注意力、VAE 与文本 encoder 权重不编辑。

标准 SD1.x K/V 输入就是文本 encoder 的序列 hidden states；有 norm_cross 时执行与原 attention processor 相同的归一化。这些输入不依赖图像 latent 和 timestep，因此拟合时不必采样来捕获相同表示。这一适配不能直接用于 DiT 深层文本状态。

新增架构分别使用：

| 架构 | 序列通路 | pooled 通路 |
| --- | --- | --- |
| SDXL base | 原生 encode_prompt 的双 CLIP penultimate hidden states 拼接，编辑 cross-attention K/V | add_embedding.linear_1 的 pooled 文本列，尺寸/裁剪列保持原值 |
| SD3.0 | 原生 CLIP 拼接并补齐特征维，再与 T5 沿 token 轴拼接；编辑 context_embedder | time_text_embed.text_embedder.linear_1，接收双 CLIP pooled 拼接 |
| FLUX.2-klein-4B | 原生 Qwen3 第 9/18/27 层表示拼接，7680 维；编辑 context_embedder | 无 pooled 文本分支 |
| FLUX.1 | 原生 T5 token hidden states，编辑 context_embedder | time_text_embed.text_embedder.linear_1，接收 CLIP pooled hidden states |

SD3/FLUX.1 当前默认两个选择单元：序列入口与 pooled 入口。Klein 只有 context_embedder 一个选择单元，both 和 tokens 均编辑该入口；pooled 会报错。开启 SU 时，Klein 当前只能判断是否可以撤去这个整体更新，不能进行 block 级筛选。它们均只接受文本条件，不修改图像输入、时间/guidance 分支或 single-stream 共享权重。图像生成仍会受到文本条件变化影响，不能据此保证图像内容完全不变。

表示由各管线自身的 encode_prompt 产生，不用 SD1.x 的 CLIP 最后一层代替其他架构的输入。SD3 的 CLIP 与 T5 分段对齐；T5 无 BOS，单独处理其首 token、EOS 和 padding。双 CLIP 的有效 token 位置需一致，但允许采用不同 padding ID，并保留各自原生的 padding 表示。

Klein 使用原生 Qwen3 chat template，参数与管线一致：add_generation_prompt=True、enable_thinking=False。token 对齐覆盖格式化后的真实输入，不将 assistant 后缀误当成 EOS；保留概念取用户正文的最后一个 token。空提示保护仍使用完整原生空提示表示。检查长度时计入 chat 包装，超长直接报错。

### 成对 CA 与 token 对齐

从原 imagenet_templates / painting_templates 选取模板：

~~~text
target: template.format(target)
anchor: template.format(anchor)
~~~

逐对提取真实 token 表示，保留上下文对应关系，不跨模板求均值。源 token 仍是实际输入行，替代 span 根据端点对应选择实际 token；公共词跨度按 token 序列对齐，EOS 和 padding 按相对 EOS 的位置对齐。

full 模式包含上下文、EOS 和默认启用的 padding 状态。SD1.x cross-attention 会消费这些 padding states，不能直接假设它们不携带概念信息。--no-include-padding 可用于消融。subject 模式仅拟合替换跨度，可能缺少后文或尾部携带的目标信息。

SDXL/SD3/FLUX 默认从 full 的候选行中，每个编码段、每个提示对选取最多约 16 个实际 token 行，优先保留替换跨度、EOS 和尾部 padding，再均匀补充上下文位置。若必要跨度超过预算则保留全部必要行；不对 hidden states 求均值。--max-fit-tokens 0 恢复全部行，会显著增加求解成本。该预算不改变生成时的文本长度。

CLIP 文本注意力具有因果性，所以仅取句首目标词不能声称包含句尾的场景；默认 full 模式同时利用序列中能读到相应上下文的位置。超出 CLIP 长度的提示会报错，避免静默截断后对齐错误。

### 零空间闭式解

以样本为列记 $X_T,X_A,X_P$，保持原公式：

$$
\min_{\Delta W=\Delta W P}
\|(W+\Delta W)X_T-WX_A\|_F^2
+\beta\|\Delta W\|_F^2,\qquad PX_P=0.
$$

求解时使用保留子空间的正交补基，在 float64 下求解缩小后的正定线性系统，不显式计算矩阵逆。相同输入空间的 K/V 复用求解，最终更新为 $W+\Delta W$，bias 保持原值。

新架构使用等价的薄 SVD 投影和低秩因子形式，按样本数与输入维度选择较小的 Gram 系统，不构造完整的 4096 维零空间基；参数更新按 $(WL)R$ 计算。与原闭式解的等价性有 CPU 数值检查。SDXL pooled 分支只将更新写入文本列，元数据列逐元素不变。

默认从本目录 instance.csv 或 style.csv 取前 128 个过滤去重后的非目标概念，保护其最后一个 subject token；另外保护完整空提示序列。可使用：

- --retain-count 0：不加入外部概念，空提示保护仍默认启用。
- --retain-count -1：使用全部保留概念，可能让保留空间满秩。
- --retain-path：替换 CSV 或逐行文本；CSV 列由 --retain-column 指定。
- --no-protect-null：关闭空提示保护，仅建议用于消融。
- --rank-rtol：数值秩容差，默认 1e-6。
- --preserve-rank：显式截断保护秩，会放松完整保留约束。

若保留空间满秩或目标没有可编辑方向，程序报错，不输出一个静默无效的“擦除成功”结论。metadata 中保存秩与数值约束诊断；这些不是图像效果指标。

### Selective update

1. 求出固定的全部候选更新。
2. 全部启用，生成校准图片，判断初始组合。
3. 每轮仅撤去一个选择单元的更新，原网络结构不变。
4. 用相同提示、seed、采样配置重新生成。
5. 任一目标重新出现时，恢复本轮更新，停止并保存上一轮组合。

默认用一次原模型轨迹上的去噪响应代理排序，时间点覆盖整个采样过程。它只决定顺序，不作为擦除成功或停止判据；不使用旧 attention map 差值或固定 Top-K。也可选择 --selection-order forward、reverse 或 random。

SDXL/SD3/FLUX 的排序在原生 pipeline 的真实 backbone 输入上取样，匹配 latent/timestep，再更换文本条件计算参考响应。它保留原生条件字段、FlowMatch scheduler 和各代 FLUX 的 latent 布局；比较的是 backbone 预测输出，作为排序代理，不是最终图像判别。Klein 输入原生 patched 4D 噪声，由管线内部生成 4D 位置编码和打包；不会套用 FLUX.1 的 3D packed 输入。采集轨迹时使用 latent 输出，不解码图片。

固定顺序最多判断 n+1 个组合。首次失败即停止对顺序敏感，不声称得到全局最小模块集合。开启 SU 时同时保存 checkpoint_all，便于中断后仍可检查全更新模型。

## 4. 可选的多模态判别接口

本版默认不调用任何接口。若已有本地多模态服务或兼容的视觉 chat API：

~~~bash
export VLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLM_MODEL="your-vision-model"
# 如果服务需要鉴权，再设置 VLM_API_KEY；不写入运行配置。

SU=1 SELECTION_JUDGE=vlm bash script/sd14_instance.sh \
  --selection-template-count 3 \
  --target-description "Snoopy: identify the specific Peanuts cartoon character, not every ordinary dog."
~~~

默认使用 /chat/completions，发送单张生成图片及目标定义，不发送含目标名的生成提示作为证据。需要视觉输入和 JSON 输出能力。返回格式为：

~~~json
{"concepts":{"Snoopy":{"present":false,"evidence":"brief visual evidence"}}}
~~~

每个目标必须显式返回 present=true / false / null。null、字段缺失、请求失败或非 JSON 都不会被接受为擦除通过。接口仅辅助 SU，不运行完整评价基准。风格判断的可靠性需要人工检查。

## 5. 输出与检查点复用

~~~text
run/
  config.json
  prompt_pairs.json
    retain_texts.json
    probe_retain_texts.json    # 与拟合保护集合分开的排序探测文本
  generation_prompts.json
  checkpoint/
    weights.safetensors       # 最终启用模块的绝对编辑权重
    metadata.json             # 基础模型、开关、rank、选择状态与模块清单
  checkpoint_all/             # 仅 SU 开启时，保存全部候选更新
  images/
    index.html                # 原图/擦除图并排浏览
    manifest.json             # 提示、模板编号、seed 与文件路径
    concept-hash/
      original/
      edit/
      compare/                # 左原图、右编辑图的合成图片
  selection/                  # 仅 SU 开启时
    history.json
    000/                      # 全更新组合的图片
    001/                      # 第一次删除试验
  summary.json
~~~

默认生成 original 与 edit 两组，使用完全相同的初始噪声。不同数量的尾批次也支持，不要求样本数能被 batch size 整除。

检查点不是完整生成骨干，也不是原 casu_code 的 .pt 格式；由本项目加载回同一个基础模型。不创建第二份 U-Net/Transformer，而是在原权重/绝对编辑权重之间切换，避免累积补丁和额外模型副本。

~~~bash
CHECKPOINT=/path/to/run/checkpoint \
CONTENTS="Snoopy,Mickey Mouse" SAMPLE_TEMPLATES=0 \
bash script/sample_sd14.sh

python main.py sample \
  --checkpoint /path/to/run/checkpoint \
  --sample-template-offset 8 --sample-template-count 8 \
  --num-samples 2 --seed 100 \
  --output-dir runs/more-images
~~~

可用 --no-generate 只保存编辑检查点，用 --modes edit 只生成编辑图。

## 6. 无模型预览与代码检查

~~~bash
bash script/sd14_instance.sh --dry-run
CA=0 bash script/sd14_style.sh --dry-run
python -m unittest discover -s tests -v
~~~

--dry-run 仅写提示和配置，不加载模型、不采样、不调用判别器。CPU 单元检查覆盖闭式解约束、token 对齐、检查点恢复和首次失败回退，不衡量真实概念擦除效果。

此前使用本地真实 SD1.4 权重，在 64×64、2 步下跑通实例 V 编辑、风格 KV 编辑、尾批次采样及检查点重载。重载后的原图和编辑图与首次生成逐像素一致。真实骨干上的代理排序及模拟人工反馈回退也已跑通；模拟反馈仅验证控制流程，不是视觉判别结果。

全部 31 项单元检查已通过，使用 Diffusers 0.37.1 的缩小随机 SDXL/SD3/FLUX.1/FLUX.2-klein 组件，覆盖编码、成对拟合、SDXL 元数据列保护、T5/Qwen3 chat 对齐、各代 FLUX latent 布局、采样/尾批次、检查点重载和原生管线代理排序。Klein 使用原生 Qwen3ForCausalLM、Flux2Transformer2DModel 和 AutoencoderKLFlux2，并验证重载后图片一致。新增检查覆盖完整评价集、CLIP 尾批次权重、FID 数值等价性、特征缓存失效、采样续跑，以及跨检查点原图复用、配置改变后覆盖、图片序号对应、任务隔离和损坏补图，并验证直接读取公共原图、清理旧链接及拒绝不匹配的原图配置。随机模型检查只验证代码流程，不评估擦除质量。

验证环境没有可用 GPU，未对完整 SDXL/SD3/FLUX 预训练大模型验证实际擦除效果，也未自动下载这些模型。请用自己的模型权重，在合适分辨率、步数下生成图片检查。

## 7. SDXL、SD3.0 和 FLUX 的启动方式

~~~bash
# 各架构均有实例和风格两条入口。
bash script/sdxl_instance.sh
bash script/sdxl_style.sh
bash script/sd3_instance.sh
bash script/sd3_style.sh
bash script/flux_instance.sh
bash script/flux_style.sh
~~~

默认 CA=1、SU=0，前 8 个模板、每个提示 1 张，输出原图、编辑图和对照网页。新架构脚本默认 1024×1024；SDXL 20 步/CFG 5，SD3 28 步/CFG 7，FLUX 默认 FLUX.2-klein-4B、4 步、guidance=1。Klein 在 CUDA 上默认开启 CPU offload，文本编码批次为 1，闭式求解仍在 CPU 上进行。效果需要自行确认，这些是初始运行配置。

~~~bash
# 本地 SDXL，K/V 联合编辑，使用所有生成模板。
SD_CKPT=/path/to/sdxl PARAMS=KV SAMPLE_TEMPLATES=0 \
bash script/sdxl_instance.sh --local-files-only

# SD3.0，同样可切换 CA/SU。
SD_CKPT=/path/to/sd3 CA=1 SU=0 CPU_OFFLOAD=1 \
TARGETS="Van Gogh" ANCHORS="realistic" \
bash script/sd3_style.sh --local-files-only

# FLUX.2-klein-4B：默认开启 CPU offload、文本编码批次 1。
SD_CKPT=/path/to/FLUX.2-klein-4B \
bash script/flux_instance.sh --local-files-only

# 显存仍紧张时，可先用较低分辨率检查流程。
HEIGHT=512 WIDTH=512 bash script/flux_style.sh

# 明确设置 CPU_OFFLOAD=0 可将模型放在 GPU；默认不这样运行 Klein。
# 兼容旧 FLUX.1：dev 为 28 步/guidance 3.5，schnell 为 4 步/guidance 0。
FLUX_VARIANT=dev bash script/flux_instance.sh
FLUX_VARIANT=schnell bash script/flux_instance.sh
~~~

默认模型 ID 分别是 stabilityai/stable-diffusion-xl-base-1.0、stabilityai/stable-diffusion-3-medium-diffusers 和 black-forest-labs/FLUX.2-klein-4B。使用模型 ID 时需要具备对应访问权限与下载环境；也可通过 SD_CKPT 指定已下载的完整 Diffusers 目录。使用本地 FLUX.1 目录时需明确 FLUX_VARIANT=dev 或 schnell。

CPU_OFFLOAD=1 使用 Diffusers 整模型 offload，仍需 CUDA；未使用 sequential offload 或量化层。dtype 默认：CPU 为 float32，CUDA 上 SDXL 为 float16，SD3/FLUX 在硬件支持时优先 bfloat16。可用 DTYPE 显式设置。

FLUX.2-klein-4B 是步数蒸馏模型，使用 guidance=1；当前适配拒绝负提示与 true CFG，避免参数被静默忽略。FLUX.1 的 embedded guidance 与普通 CFG 不同；schnell 必须 guidance=0。仅在 FLUX.1 使用负提示时，需显式 --true-cfg-scale > 1，例如 TRUE_CFG_SCALE=2，并传入 --negative-prompt，它会增加 Transformer 调用次数。

所有新脚本继续接受 CA、SU、TARGETS、ANCHORS、OUTPUT_DIR 等环境变量及额外 CLI 参数：

~~~bash
CA=0 SU=0 bash script/sdxl_instance.sh
CA=1 SU=1 SELECTION_ORDER=proxy bash script/sd3_instance.sh

# 单通路消融与拟合规模。
BRANCHES=tokens MAX_FIT_TOKENS=8 bash script/flux_instance.sh
BRANCHES=pooled bash script/sdxl_style.sh

# 仅检查配置和提示，不加载模型。
bash script/flux_style.sh --dry-run
~~~

对于新检查点，使用统一采样脚本，架构和 FLUX 变体从检查点读取：

~~~bash
CHECKPOINT=/path/to/run/checkpoint CPU_OFFLOAD=1 \
bash script/sample_model.sh

python main.py erase --architecture sd3 \
  --task instance --target-concepts Snoopy --anchor-concepts dog \
  --contextual-anchor --no-selective-update \
  --model-cpu-offload --sample-template-count 8

python main.py sample --checkpoint /path/to/run/checkpoint \
  --model-cpu-offload --sample-template-count 8
~~~

参考实现与参数语义：[FLUX.2-klein-4B 模型说明](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)、[Diffusers 0.37.1 Klein 管线](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)。本项目固定 Diffusers 0.37.1。

## 8. 完整实验集与 CLIP / FID

复用当前采样实现，给定已经擦除的检查点即可连续完成两步：

~~~bash
CHECKPOINT=/path/to/erasure-run/checkpoint \
bash script/sample_and_metrics.sh
~~~

任务从检查点读取。instance 默认生成五个角色 × 全部 80 模板 × 每提示 10 张，即 original/edit 各 4000 张；style 默认三位画家 × 全部 30 模板 × 每提示 10 张，即各 900 张。默认沿用检查点生成配置，支持 SD1.x、SDXL、SD3 和 FLUX。采用相同初始噪声比较原模型与编辑模型。

CLIP 使用旧评价脚本 CLI 默认的 openai/clip-vit-base-patch16，计算图片与完整生成提示的 100×cosine 均值。FID 沿用 torch-fidelity 的 inception-v3-compat / 2048，按概念计算 original 与 edit 的分布差异。结果默认保存在 metric/<擦除运行目录名>/results.json、per_concept.csv、per_image_clip.csv，含 target/non_target 分组。

支持断点续跑；采样结束后释放生成模型，再启动指标进程。完整配置、CSV 采样、独立指标命令，以及原评价脚本的问题分析，见 [metric/README.md](metric/README.md)。

original 根据检查点架构自动选择原图根目录，不随 PAPER_RUN_ROOT 或 IMAGE_DIR 改变，不使用文件锁或 hash 目录：

| 模型架构 | 默认原图根目录 |
| --- | --- |
| SD1.x（含 SD1.4） | runs/original/ |
| SDXL | runs/original_SDXL/ |
| SD3.0 | runs/original_SD3/ |
| FLUX | runs/original_FLUX/ |

各根目录内仍按 <任务>/<概念>/ 保存，因此切换到 SDXL、SD3.0 或 FLUX 不会覆盖 SD1.4 的原图。单独采样命令和 paper 批量脚本均自动使用该设置，无需手动设置 ORIGINAL_DIR。以 SD1.4 为例：

~~~text
runs/original/
  instance/snoopy/
    config.json       # 格式化的采样配置
    manifest.json     # 序号文件名、完整 prompt、seed、模板及样本编号
    000000.png
    000001.png
  instance/mickey/
  style/van_gogh/
  style/monet/
  style/picasso/
~~~

直接读取该概念的 config.json 和 manifest.json，与本次参数、提示及 seed 清单比较。两者一致时跳过已有有效图片；不一致时删除该模型、概念目录下的旧 PNG，在同一目录重新生成并覆盖 config/manifest。config.json 使用缩进排版。CA、SU、anchor、擦除目标与编辑权重不属于 original 的配置，因此只修改擦除参数可以继续复用。同一架构内若要同时保留不同基础模型或不同采样配置的原图，可分别指定 ORIGINAL_DIR；显式指定的目录优先于默认目录。

实验目录保存 edit 图片及 manifest，manifest 的 original 字段直接记录公共原图的绝对路径，例如 /home/shx/code/ce-new/casu-2/runs/original/instance/snoopy/000000.png。默认不再创建实验内的 original 子目录，不复制图片、不创建硬链接或软链接。对照网页和可选拼图也直接读取公共原图路径。CLIP/FID 使用同一条记录的完整 prompt 和 original/edit 路径，读取外部原图时核对公共目录的 config/manifest。公共原图若已被不同采样参数覆盖，旧实验计算指标时会提示配置不符。

对已有实验重新运行采样脚本，会复用有效的公共原图、续跑 edit，并在成功更新 manifest 后删除本实验清单对应的旧 original 文件及空目录；不删除公共原图和 edit。双卡任务仍由 shell 分配，当前 instance 在 GPU 0、style 在 GPU 1；不对同时写入同一原图目录的进程做协调。旧 hash 目录不再读取。

~~~bash
# 原有命令即可自动复用原图；改变擦除参数时使用新的实验目录。
PAPER_RUN_ROOT=/path/to/new-erasure-experiment \
NUM_GPUS=2 bash script/paper_sample_all.sh

# 可将原图保存到指定磁盘，后续实验沿用相同路径。
ORIGINAL_DIR=/path/to/original bash script/paper_sample_all.sh

# 关闭公共原图目录，恢复逐实验生成原图（同目录续跑仍跳过已有图片）。
ORIGINAL_CACHE=0 bash script/paper_sample_all.sh
~~~

CLI 对应 --original-dir /path/to/original 和 --no-original-cache；旧的 ORIGINAL_CACHE_DIR / --original-cache-dir 保留为目录参数别名。修改擦除权重后仍需新的实验图片目录，以保证 edit 和指标来自当前检查点；相同配置的 original 会继续复用。

## 9. 原论文的六组擦除实验

新增六组独立擦除和对应的完整实验集采样脚本。默认模型为 SD1.4（CompVis/stable-diffusion-v1-4），CA=1、SU=0，512×512、20 步、CFG 7.5；可用 ARCHITECTURE=sdxl、sd3、flux 切换架构，SD_CKPT 指定原始基础模型。各组都从同一原始基础模型重新加载，二/三目标在一次闭式求解中同时擦除，不沿用上一组的编辑权重。

~~~bash
cd /home/shx/code/ce-new/casu-2

# 先独立完成六组擦除，只保存检查点；开启 SU 时仍生成校准图。
bash script/paper_erase_all.sh

# 再逐组生成完整实验集，并计算 CLIP/FID。
bash script/paper_sample_all.sh

# 或一次执行上述两个阶段。
bash script/paper_all.sh all
~~~

各组也有单独入口：

| 擦除目标 | 擦除脚本 | 对应生成脚本 |
| --- | --- | --- |
| Snoopy | script/paper_erase_instance_snoopy.sh | script/paper_sample_instance_snoopy.sh |
| Snoopy、Mickey | script/paper_erase_instance_snoopy_mickey.sh | script/paper_sample_instance_snoopy_mickey.sh |
| Snoopy、Mickey、Spongebob | script/paper_erase_instance_snoopy_mickey_spongebob.sh | script/paper_sample_instance_snoopy_mickey_spongebob.sh |
| Van Gogh | script/paper_erase_style_van_gogh.sh | script/paper_sample_style_van_gogh.sh |
| Monet | script/paper_erase_style_monet.sh | script/paper_sample_style_monet.sh |
| Picasso | script/paper_erase_style_picasso.sh | script/paper_sample_style_picasso.sh |

生成脚本固定评价完整的五个角色或三位画家。每组 instance 为 original/edit 各 4000 张，每组 style 各 900 张；六组目录合计引用 29400 张图片，其中相同配置的 original 共用该架构的公共原图目录。默认包含 CLIP/FID；WITH_METRICS=0 时仅生成图片。六组分别计算指标，不能将它们合并当成同一次擦除实验。

默认路径按模型与实验组区分：

~~~text
runs/paper-sd1/
  erase/<组名>/checkpoint/
  erase/<组名>/checkpoint_path.txt
  images/<组名>/
  plans/<组名>/             # 仅 DRY_RUN=1 的擦除计划
metric/paper-sd1/<组名>/
~~~

生成脚本读取对应组的 checkpoint_path.txt；没有指针时读取该组 checkpoint/，并核对目标和任务。可用 PAPER_RUN_ROOT、PAPER_IMAGE_ROOT、PAPER_METRIC_ROOT 自定义各阶段目录。指标根目录默认使用 metric/<PAPER_RUN_ROOT 目录名>/。切换模型或 CA/SU 等实验配置时，使用新的 PAPER_RUN_ROOT；擦除输出非空时不会自动覆盖，采样则默认按原有配置安全续跑。

~~~bash
# 使用本地 Klein 模型，固定两阶段共用的实验根目录。
ARCHITECTURE=flux SD_CKPT=/path/to/FLUX.2-klein-4B \
PAPER_RUN_ROOT=/path/to/experiments/klein-ca1-su0 \
LOCAL_FILES_ONLY=1 bash script/paper_erase_all.sh

ARCHITECTURE=flux PAPER_RUN_ROOT=/path/to/experiments/klein-ca1-su0 \
LOCAL_FILES_ONLY=1 METRIC_BATCH_SIZE=8 bash script/paper_sample_all.sh

# 只生成六组图片，暂不计算指标。
WITH_METRICS=0 bash script/paper_sample_all.sh

# 默认 SD1.4，无需额外选择架构。
bash script/paper_erase_all.sh
bash script/paper_sample_all.sh

# 无需模型的六组擦除计划检查，写入 plans/，不占用真实擦除目录。
DRY_RUN=1 bash script/paper_erase_all.sh
~~~

instance 替代概念默认分别为 dog、mouse、cartoon character，风格替代描述默认 realistic。可通过 SNOOPY_ANCHOR、MICKEY_ANCHOR、SPONGEBOB_ANCHOR、STYLE_ANCHOR 修改，包括显式设置为空。CA、SU、CPU_OFFLOAD、DEVICE、DTYPE、HEIGHT、WIDTH、SEED 等继续有效。擦除始终在一张卡上逐组执行；生成默认单卡，也可用两张卡各自运行一个实验组。单卡任务遇错停止；双卡时某组失败会停止该组的后续任务，等待另一组结束后返回非零状态。若 SU 启用，仍需按既有 manual/VLM 判别流程完成校准。

paper_erase_all.sh 的额外参数传给擦除 CLI；paper_sample_all.sh 的额外参数传给采样 CLI。paper_all.sh all 使用环境变量配置两个阶段。单组脚本不接受覆盖其固定目标、任务和输出路径的 CLI 参数，以避免组名与实际实验不一致。

## 10. 擦除单卡执行，生成选择一张或两张 GPU

GPU 直接在 shell 脚本中通过 CUDA_VISIBLE_DEVICES 指定，没有 Python GPU 调度器。擦除始终单卡逐组执行。生成脚本 script/paper_sample_all.sh 默认 NUM_GPUS=1；设为 2 时，同时启动下面两段固定的任务列表，每张卡内依次执行，每张卡同时只有一个生成实验：

| shell 中的指定 | 依次运行的生成实验 |
| --- | --- |
| export CUDA_VISIBLE_DEVICES=0 | Snoopy；Snoopy+Mickey+Spongebob；Snoopy+Mickey |
| export CUDA_VISIBLE_DEVICES=1 | Van Gogh；Monet；Picasso |

需要换卡时，直接修改这两行的 0、1；需要改变实验分配时，把对应的 bash 命令移动到另一段。双卡分支中的这两行会覆盖启动时继承的 CUDA_VISIBLE_DEVICES。单卡分支则沿用启动时设置的 CUDA_VISIBLE_DEVICES，默认 0。DEVICE 保持默认 cuda，指向该进程唯一可见的显卡。

~~~bash
cd /home/shx/code/ce-new/casu-2

# 擦除：始终单卡逐组执行。
CUDA_VISIBLE_DEVICES=0 bash script/paper_erase_all.sh

# 单卡生成：六组依次在 GPU 0 上运行。
CUDA_VISIBLE_DEVICES=0 bash script/paper_sample_all.sh

# 单卡生成：改用 GPU 1。
CUDA_VISIBLE_DEVICES=1 bash script/paper_sample_all.sh

# 双卡生成：使用 shell 双卡分支中写明的 GPU 0、1。
NUM_GPUS=2 bash script/paper_sample_all.sh

# 一键执行：先在 GPU 0 上完成全部擦除，再在 GPU 0、1 上并行生成。
NUM_GPUS=2 bash script/paper_all.sh all

# 无 GPU 也可检查六组擦除计划。
DRY_RUN=1 CUDA_VISIBLE_DEVICES=0 bash script/paper_erase_all.sh
~~~

也可以手动指定两个生成实验及其 GPU，适合只运行选定的组合：

~~~bash
# 各组对应的擦除检查点须已生成。
CUDA_VISIBLE_DEVICES=0 bash script/paper_sample_instance_snoopy.sh &
pid_snoopy=$!
CUDA_VISIBLE_DEVICES=1 bash script/paper_sample_instance_snoopy_mickey.sh &
pid_snoopy_mickey=$!
wait "$pid_snoopy"
wait "$pid_snoopy_mickey"
~~~

上述两组沿用论文已有的 Snoopy 与 Snoopy+Mickey 设置。每组默认生成 original/edit 的完整评价图片，并计算该组 CLIP/FID；WITH_METRICS=0 可只生成图片。指标进程继承该组的 CUDA_VISIBLE_DEVICES，METRIC_DEVICE=cpu 可将指标放在 CPU。

all 模式先完成单卡串行擦除，再调用生成脚本；NUM_GPUS 只影响生成阶段。不论 SU 关闭、manual 或 VLM 判别，擦除阶段都只有一个任务。擦除可在 script/paper_erase_all.sh 中修改默认显卡，或在启动时设置 CUDA_VISIBLE_DEVICES。

输出路径继续按实验组区分，单卡和双卡使用相同的数据集计划与 seed；切换 CUDA 显卡编号可复用已完成图片，仍检查检查点、提示、分辨率、步数等设置。双卡输出直接显示在终端，Ctrl-C 或 SIGTERM 会停止两个生成任务组及其子进程。

每个任务的 CPU_THREADS 默认仍为 8，双卡生成时可按服务器资源调小，例如 CPU_THREADS=4。两张卡分别加载自己的生成模型，主存占用也会增加。

默认模型仍为 SD1.4，ARCHITECTURE 切换方式保持不变。
