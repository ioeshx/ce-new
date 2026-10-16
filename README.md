# CASU-2：SD 1.x、SDXL、SD3.0 与 FLUX 概念擦除

独立实现于本目录，不修改、不导入 aaai27/casu_code 的源码。支持闭式擦除、检查点保存、原模型与编辑模型的同噪声采样，以及 CLIP / FID 评价。

统一 Python 入口为 `main.py erase`、`main.py sample` 和 `main.py metrics`。Shell 只保留五个脚本：

```text
script/
  erase.sh             # 设置擦除参数，直接调用 main.py erase
  sample.sh            # 设置采样和指标参数，直接调用 main.py sample / metrics
  paper_erase_all.sh   # 依次调用 erase.sh，完成六组擦除
  paper_sample_all.sh  # 调用 sample.sh，单卡顺序运行或双卡分组运行
  paper_all.sh         # 选择 erase、sample 或先擦除再采样
```

参数直接在 `erase.sh`、`sample.sh` 开头修改，也可追加命令行参数；脚本不读取环境变量作为实验配置。各架构的默认模型和采样设置由 `models.py` 统一处理。凭据不属于实验配置，VLM API 密钥仍从指定的密钥环境变量读取。

## 1. 安装与首次运行

建议 Python 3.10+，在独立环境中安装：

```bash
cd /home/shx/code/ce-new/casu-2
python -m pip install -r requirements.txt

# 擦除 Snoopy → dog，只保存编辑检查点。
bash script/erase.sh instance_snoopy

# 从对应检查点生成完整评价集，再计算 CLIP / FID。
bash script/sample.sh instance_snoopy
```

默认基础模型为 SD1.4，512×512、20 步、CFG 7.5。擦除脚本默认开启 contextual anchor、关闭 selective update。采样脚本默认每个提示 10 张，使用完整模板集，并开启续跑和公共原图复用。

在 `erase.sh` 开头设置 `ARCHITECTURE`、`MODEL`、`DEVICE`、`SEED`、`PARAMS`、`BETA`、`RETAIN_COUNT` 和各目标的 anchor；`ERASE_OPTIONS` 数组可添加其他 Python 参数。`MODEL` 留空使用架构默认模型，指定本地模型时填写完整 Diffusers 目录。

在 `sample.sh` 开头设置 `ARCHITECTURE`、`DEVICE`、`SEED`、`SAMPLE_TEMPLATES`、`NUM_SAMPLES`、`BATCH_SIZE`、`WITH_METRICS` 和指标参数。分辨率、步数和 guidance 默认从检查点读取，也可放进 `SAMPLE_OPTIONS` 数组。

`RUN_ROOT` 留空时按架构使用 `runs/paper-sd1`、`runs/paper-sdxl`、`runs/paper-sd3` 或 `runs/paper-flux-<变体>`。修改架构或实验根目录时，两份脚本使用一致的设置。所有入口可从其他工作目录调用；默认数据和输出定位到本项目，相对的自定义路径按当前工作目录解析。

```bash
# 临时覆盖 seed，不必修改环境变量。
bash script/erase.sh instance_snoopy --seed 42
bash script/sample.sh instance_snoopy --seed 42

# 使用本地模型，在新目录中完成一次独立实验。
bash script/erase.sh instance_snoopy \
  --sd-ckpt /path/to/sd14 --local-files-only --run-root /path/to/new-run
bash script/sample.sh instance_snoopy \
  --local-files-only --run-root /path/to/new-run
```

擦除目录非空时默认拒绝覆盖，确需覆盖可显式追加 `--overwrite`。更换擦除权重或采样配置时使用新的实验目录。

## 2. CA 与 selective update

直接修改 `erase.sh` 中的 `ERASE_OPTIONS`，或在调用时追加对应参数：

```bash
bash script/erase.sh instance_snoopy --contextual-anchor --no-selective-update
bash script/erase.sh instance_snoopy --no-contextual-anchor --no-selective-update
bash script/erase.sh instance_snoopy --contextual-anchor --selective-update
bash script/erase.sh instance_snoopy --no-contextual-anchor --selective-update
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| --contextual-anchor / --no-contextual-anchor | 开启 | 成对上下文拟合 / 独立概念提示拟合 |
| --selective-update / --no-selective-update | 关闭 | 启用 / 关闭生成反馈的逐模块删除 |
| --selection-judge manual / vlm | manual | 人工判定 / 多模态接口判定 |
| --params V / K / KV | V | SD1 / SDXL 的文本条件投影 |
| --token-mode full / subject | full | 全文本序列 / 替换 token 跨度 |
| --condition-branches both / tokens / pooled | both | 序列与 pooled 条件通路 |
| --max-fit-tokens | 16 | 新架构每段、每对提示的拟合行预算；0 使用全部 |

其他参数用 `python main.py erase --help` 查看。追加的普通参数覆盖脚本默认值，不再限制覆盖目标、任务或路径。

开启 selective update 会增加生成次数。manual 模式在每轮打印图片网页路径并等待输入：`no` 表示均未见待擦除目标，接受删除；`yes` 表示目标出现，恢复更新并停止；`unknown` 表示无法确认，停止并保留上一轮。判别失败或缺少输入不会自动算作通过。

自定义目标也可直接调用 Python：

```bash
python main.py erase --task instance \
  --target-concepts "Snoopy,Mickey Mouse" --anchor-concepts "dog,mouse" \
  --contextual-anchor --no-selective-update --no-generate \
  --output-dir runs/custom-erasure
```

可以给所有目标一个共享 anchor，或按顺序给出等量 anchors。空 anchor 可用 `--anchor-concepts ""`。anchor 需要根据概念范围选择，默认示例没有保证是最优替代概念。

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

求解设备默认跟随 `--device`，例如 `--device cuda:1` 会在 GPU 1 上求解。可用 `--solve-device cpu` 将求解单独放到 CPU，模型加载与生成设备不受影响。

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
# 如果服务需要鉴权，凭据仍通过 VLM_API_KEY 提供，不写入运行配置。
bash script/erase.sh instance_snoopy \
  --selective-update --selection-judge vlm \
  --judge-base-url "http://127.0.0.1:8000/v1" \
  --judge-model "your-vision-model" --selection-template-count 3 \
  --target-description "Snoopy: identify the specific Peanuts cartoon character, not every ordinary dog."
~~~

默认使用 /chat/completions，发送单张生成图片及目标定义，不发送含目标名的生成提示作为证据。需要视觉输入和 JSON 输出能力。返回格式为：

~~~json
{"concepts":{"Snoopy":{"present":false,"evidence":"brief visual evidence"}}}
~~~

每个目标必须显式返回 present=true / false / null。null、字段缺失、请求失败或非 JSON 都不会被接受为擦除通过。接口仅辅助 SU，不运行完整评价基准。风格判断的可靠性需要人工检查。

## 5. 输出与检查点复用

```text
runs/paper-sd1/
  erase/<实验名>/
    config.json
    prompt_pairs.json
    retain_texts.json
    probe_retain_texts.json
    generation_prompts.json
    checkpoint/
      weights.safetensors
      metadata.json
    checkpoint_all/       # selective update 开启时
    selection/            # selective update 开启时
    summary.json
  images/<实验名>/
    config.json
    dataset.json
    generation_prompts.json
    manifest.json
    index.html
metric/paper-sd1/<实验名>/
  results.json
  per_concept.csv
  per_image_clip.csv
  cache/
```

检查点只保存本项目的局部编辑权重及元数据，需要加载对应的原始基础模型。采样在原权重与编辑权重间切换，不另建一份模型。默认 original 与 edit 使用相同提示、seed 和初始噪声。

需要采样其他检查点或输出到其他目录时：

```bash
bash script/sample.sh instance_snoopy \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/images --metric-dir /path/to/metrics

# 仅生成图片。
bash script/sample.sh instance_snoopy --no-metrics

# 跳过采样，只复算已有图片的指标。
bash script/sample.sh instance_snoopy --metrics-only
```

Shell 会将 `--checkpoint`、`--output-dir` 和设备同时用于相应阶段；其余 Python 采样参数透传给 `main.py sample`。指标参数在 `sample.sh` 开头或 `METRIC_OPTIONS` 数组设置。`--local-files-only` 同时作用于采样和指标模型。

## 6. 命令与脚本检查

```bash
python main.py erase --help
python main.py sample --help
python main.py metrics --help
bash -n script/erase.sh
bash -n script/sample.sh
bash -n script/paper_erase_all.sh
bash -n script/paper_sample_all.sh
```

这些命令检查入口和 Shell 语法，不会启动实验。

## 7. SDXL、SD3.0 和 FLUX

在 `erase.sh` 和 `sample.sh` 开头修改 `ARCHITECTURE`，或使用命令行参数：

```bash
bash script/erase.sh instance_snoopy --architecture sdxl
bash script/sample.sh instance_snoopy --architecture sdxl

bash script/erase.sh style_van_gogh --architecture sd3 --model-cpu-offload
bash script/sample.sh style_van_gogh --architecture sd3

bash script/erase.sh instance_snoopy --architecture flux --flux-variant klein
bash script/sample.sh instance_snoopy --architecture flux --flux-variant klein
```

| 架构 | 默认模型 | 分辨率 | 步数 | guidance |
| --- | --- | --- | --- | --- |
| sd1 | CompVis/stable-diffusion-v1-4 | 512×512 | 20 | 7.5 |
| sdxl | stabilityai/stable-diffusion-xl-base-1.0 | 1024×1024 | 20 | 5 |
| sd3 | stabilityai/stable-diffusion-3-medium-diffusers | 1024×1024 | 28 | 7 |
| flux / klein | black-forest-labs/FLUX.2-klein-4B | 1024×1024 | 4 | 1 |
| flux / dev | black-forest-labs/FLUX.1-dev | 1024×1024 | 28 | 3.5 |
| flux / schnell | black-forest-labs/FLUX.1-schnell | 1024×1024 | 4 | 0 |

Python 统一解析这些默认值，Shell 不再维护另一套模型分支。Klein 在 CUDA 上默认开启模型 CPU offload，文本编码批次默认 1，最大文本长度 512；其他架构文本编码批次默认 4。CPU 默认 float32，CUDA 上 SDXL 默认 float16，SD3 / FLUX 在支持时默认 bfloat16。

使用本地模型可设置 `erase.sh` 的 `MODEL`，或追加 `--sd-ckpt /path/to/model`。采样从检查点读取原始模型与架构，不需再指定模型路径。FLUX.1 使用 `--flux-variant dev` 或 `schnell`；Klein 使用 guidance=1，不能添加负提示或 true CFG。

## 8. 完整实验集与 CLIP / FID

`sample.sh` 默认先执行 `main.py sample`，生成进程结束后再执行 `main.py metrics`，释放生成模型占用的显存。可在顶部设置 `WITH_METRICS=0` 关闭指标，或设置 `SKIP_SAMPLE=1` 仅复算指标。

| 任务 | 评价概念 | 模板/概念 | 图片/提示 | original / edit 各自图片数 |
| --- | --- | --- | --- | --- |
| instance | Hello Kitty、Mickey、Pikachu、Snoopy、Spongebob | 80 | 10 | 4000 |
| style | Van Gogh、Monet、Picasso | 30 | 10 | 900 |

CLIP 使用 `openai/clip-vit-base-patch16`，计算图片与完整生成提示的 100×cosine 均值。FID 使用 torch-fidelity 的 inception-v3-compat / 2048，逐概念比较 original 与 edit。结果含目标 / 非目标分组，详细定义见 [metric/README.md](metric/README.md)。

公共原图按检查点架构分别保存在 `runs/original/`、`runs/original_SDXL/`、`runs/original_SD3/`、`runs/original_FLUX/`，每个根目录下按任务 / 概念保存。配置和提示相同时复用有效 PNG，改变采样配置时重新生成。实验 manifest 直接记录公共原图路径，编辑图片保存在本实验目录。

```bash
# 自定义公共原图目录。
bash script/sample.sh instance_snoopy --original-dir /path/to/original

# 使用独立原图，不复用公共目录。
bash script/sample.sh instance_snoopy --no-original-cache

# 按指定 CSV 提示生成。
bash script/sample.sh instance_snoopy --prompt-csv data/ip_imagenet_prompts.csv
```

修改擦除权重后使用新的实验图片目录；相同配置的公共原图仍可复用。双卡按任务分组，避免同时写入同一任务 / 概念的公共目录。

## 9. 六组批量实验

两个批量脚本分别直接调用 `erase.sh` 和 `sample.sh`，不经过其他运行脚本：

```bash
bash script/paper_erase_all.sh
bash script/paper_sample_all.sh
# 或依次完成全部擦除与采样。
bash script/paper_all.sh all
```

| 实验名 | 擦除目标 | 默认 anchor |
| --- | --- | --- |
| instance_snoopy | Snoopy | dog |
| instance_snoopy_mickey | Snoopy、Mickey | dog、mouse |
| instance_snoopy_mickey_spongebob | Snoopy、Mickey、Spongebob | dog、mouse、cartoon character |
| style_van_gogh | Van Gogh | realistic |
| style_monet | Monet | realistic |
| style_picasso | Picasso | realistic |

单组运行：`bash script/erase.sh 实验名`、`bash script/sample.sh 实验名`。每组从原始基础模型开始，多个目标在同一次擦除中处理，六组之间不累积更新。

`paper_all.sh erase` / `sample` 将额外参数传给对应阶段，`all` 将参数传给两个阶段，因此只追加两阶段都支持的参数，例如：

```bash
bash script/paper_all.sh all --architecture sdxl --seed 42 --run-root /path/to/experiment
```

阶段专用选项在对应脚本顶部修改，或分阶段传入。

## 10. GPU 分配

GPU 通过 Python 的 `--device` 参数传递，不使用 Shell 环境变量配置。单组脚本顶部的 `DEVICE="cuda:0"` 控制默认设备，也可追加 `--device cuda:1`。

批量擦除的设备由 `paper_erase_all.sh` 顶部 `DEVICE` 指定，六组依次执行。

批量生成修改 `paper_sample_all.sh` 顶部：

```bash
NUM_GPUS=1
DEVICE0="cuda:0"
DEVICE1="cuda:1"
```

`NUM_GPUS=1` 时六组依次使用 `DEVICE0`；改为 `2` 时启动两组后台任务：

| 设备变量 | 组内依次运行的实验 |
| --- | --- |
| DEVICE0 | Snoopy；Snoopy+Mickey+Spongebob；Snoopy+Mickey |
| DEVICE1 | Van Gogh；Monet；Picasso |

两组并行、每组内部顺序执行，指标默认使用与采样相同的设备。可在 `sample.sh` 设置 `METRIC_DEVICE="cpu"` 或追加 `--metric-device cpu`。中断时停止后台任务组；任一组失败会停止该组后续实验，等待另一组结束后返回非零状态。

`paper_all.sh all` 先完成全部擦除，再启动生成；双卡设置仅影响生成阶段。
