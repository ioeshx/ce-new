# CASU-2：SD 1.x、SDXL、SD3.0 与 FLUX 概念擦除

独立实现于本目录，不修改、不导入 aaai27/casu_code 的源码。支持闭式擦除、检查点保存、原模型与编辑模型的同噪声采样，以及 CLIP / FID 评价。

统一 Python 入口为 `main.py erase`、`main.py sample` 和 `main.py metrics`。主要脚本：

```text
script/
  erase.sh             # 设置擦除参数，直接调用 main.py erase
  sample.sh            # 设置采样和指标参数，直接调用 main.py sample / metrics
  paper_erase_all.sh   # 依次调用 erase.sh，完成六组擦除
  paper_sample_all.sh  # 调用 sample.sh，单卡顺序运行或双卡分组运行
  paper_all.sh         # 选择 erase、sample 或先擦除再采样
  paper_su_middle.sh   # SD1 / SDXL，CA + 中间 50% 注意力层 SU
  paper_su_vlm.sh      # 各模型，CA + 实际可编辑分组的本地 VLM 候选搜索
  paper_su.sh          # 上面两份 SU 脚本的公共参数分发，支持 --architecture
```

参数直接在 `erase.sh`、`sample.sh` 开头修改，也可追加命令行参数；脚本不读取环境变量作为实验配置。各架构的默认模型和采样设置由 `models.py` 统一处理。凭据不属于实验配置，VLM API 密钥仍从指定的密钥环境变量读取。

四种架构的擦除流程共用同一套实现：

| 文件 | 职责 |
| --- | --- |
| `models.py` | 模型加载、默认配置和架构分派 |
| `sd.py`、`sdxl.py`、`sd3.py`、`flux.py` | 定义各架构的编辑位置与分组 |
| `conditioning.py` | 原生文本编码、token 对齐、输入矩阵构造和编辑权重组装 |
| `solver.py` | 公共闭式求解；运行时统一调用 `solve_factors()` |
| `editing.py` | 公共 K/V 工具、原始/编辑权重切换和检查点读写 |
| `selection.py` | 统一的中间层选择、候选构造、分阶段 VLM 检测、旧逐层删除 |

SD1 使用自己的 CLIP 最后一层输出，保留完整 token 对齐行，不应用 SDXL/SD3/FLUX 的拟合 token 预算。四种架构的检查点继续使用 `casu2-<架构>-v1` 格式，原有检查点可继续加载。

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

默认基础模型为 SD1.4，512×512、20 步、CFG 7.5。擦除脚本默认开启 contextual anchor、关闭 selective update。采样脚本默认每个提示 10 张，使用完整模板集，并开启续跑、公共原图复用和左右对照图保存。

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
| --selective-update / --no-selective-update | 关闭 | 启用 / 关闭选择性更新 |
| --selection-method middle / vlm / legacy | SD1 为 middle，其他架构为 legacy | 中间层 / VLM 候选搜索 / 旧逐层删除 |
| --su-min-fraction / --su-max-fraction | 0.5 / 0.75 | 候选使用 ceil(min × N)..floor(max × N) 个分组 |
| --selection-judge manual / vlm | manual | 旧逐层删除流程：人工判定 / 多模态接口判定 |
| --params V / K / KV | V | SD1 / SDXL 的文本条件投影 |
| --token-mode full / subject | full | 全文本序列 / 替换 token 跨度 |
| --condition-branches both / tokens / pooled | both | 序列与 pooled 条件通路 |
| --max-fit-tokens | 16 | 新架构每段、每对提示的拟合行预算；0 使用全部 |

其他参数用 `python main.py erase --help` 查看。追加的普通参数覆盖脚本默认值，不再限制覆盖目标、任务或路径。

所有模型使用同一份 `selection.py`，不再有 `sd1_selection.py`。统一参数为 `--selection-method`；旧的 `--sd1-su-method` 仅作为兼容别名。为兼容旧命令，未指定选择方法时 SD1 默认 `middle`，其他架构仍默认 `legacy`；两份新 SU 脚本会显式选择 `middle` 或 `vlm`。

`middle` 直接选层并保存权重，无需输入，也不加载 VLM。`legacy` 的 manual 判别在每轮打印图片网页路径并等待输入：`no` 表示均未见待擦除目标，接受删除；`yes` 表示目标出现，恢复更新并停止；`unknown` 表示无法确认，停止并保留上一轮。

### 统一 SU：模型分组与适用范围

| 模型 | middle | vlm 的选择单元 |
| --- | --- | --- |
| SD1.x | 支持，U-Net 中间注意力层 | 同一组有深度顺序的注意力层 |
| SDXL | 支持，U-Net 中间注意力层 | 同一组有深度顺序的注意力层；全局 pooled 入口不参与此规则 |
| SD3.0 | 当前入口适配器没有中间 block 分组，明确报错 | `context_embedder` 和 `pooled_text` 两个入口分组 |
| FLUX.1 | 同 SD3 | 序列文本与 pooled 文本入口两个分组 |
| FLUX.2-klein | 当前仅一个入口，没有中间层分组 | 一个 `context_embedder`，需显式 `--su-max-fraction 1` |

对 SD1/SDXL，两种方法以**一个 `attn2` 跨注意力层为选择单元**；`--params KV` 时同层 K/V 一起选，不分别计数。先按 U-Net 实际顺序 `down_blocks → mid_block → up_blocks` 排序，以 `mid_block` 为中心，按距离从近到远选取；等距离时先选 up 一侧。不会直接使用 `named_modules()` 的注册顺序。这里的“中间”是 U-Net 瓶颈附近的拓扑位置，属于经验性选择规则。SDXL 的 `add_embedding.text` 是没有深度位置的全局入口，`middle` 和按中间层扩展的 `vlm` 都保持它的原始权重；排除项写入 `layer_plan.json`。

- `middle`：固定选 `ceil(0.5 × N)` 层。SD1.4 有 16 个 `attn2`，因此只更新 8 层。其余层保持原权重；不运行校准生成或 VLM。
- `vlm`：U-Net 从同一组 50% 中间层开始，每次加入距 mid 最近的一个剩余层，直到 75%。SD1.4 的候选为 8、9、10、11、12 层。每组从同一原始权重应用绝对更新，使用相同提示和噪声，不累加前一候选的更新。

对于只有并行文本入口的适配器，VLM 按实际入口分组构造候选，不将入口冒充为中间 block。以 SD3 的两个入口为例，默认 50%..75% 的整数范围只有 1，所以分别测试“仅序列入口”和“仅 pooled 入口”；若两者都无法擦除，搜索失败，默认不突破 75% 上限。可显式 `--su-max-fraction 1` 让双入口更新也成为候选。单入口模型同样需要显式上限 1，这种情况下没有减少入口数量，只能校准此更新是否通过。一般并行入口的候选按适配器声明顺序构造循环扩展链并去重，最多 O(N²) 个组合，不枚举 2^N 个集合；同样不保证全局最优组合。

VLM 校准默认读取新增的 `data/su_prompts.json`，与 CA 拟合和正式评估的模板集分开。默认每个目标 3 个提示、每提示 1 张图；SD1.4 共 5 组候选，即每目标生成 15 张校准图。可通过 `--selection-template-count`（默认文件最多 6）、`--selection-samples` 或 `--su-prompts /path/to/prompts.json` 修改。正式 CLIP/FID 采样仍使用原论文评估集，不使用校准图。

先生成**全部候选**的校准图，再删除 SD pipeline 和持有参数引用的 controller，回收对象、清空 CUDA 缓存，然后加载本地 VLM。默认模型 `Qwen/Qwen2.5-VL-3B-Instruct`，CUDA 上使用 FP16，单图检测，检测输入最长边 448。使用已有 Transformers 依赖，不需要启动常驻的 VLM 服务；首次使用需有模型文件或允许下载。用 `--su-vlm-model /path/to/vlm` 可以更换本地模型或模型 ID。

按更新分组数从少到多检测，每张图检查所有待擦除概念；**全部图片、全部目标都明确为 absent** 时才通过，首个通过候选即最终选择。任一目标出现则该候选失败，无法判断/无效 JSON 也不算通过。如果直到上限都没有通过，退出码为 2，保留校准图和候选权重，但不发布 `checkpoint/`，后续采样不会误用未通过的候选。VLM 关闭并释放显存后，才会按需重新加载对应扩散模型生成最终对照图。

```bash
# 各自完成六组擦除，再生成完整评估集并计算 CLIP / FID；默认开启 CA。
bash /home/shx/code/ce-new/casu-2/script/paper_su_middle.sh all
bash /home/shx/code/ce-new/casu-2/script/paper_su_vlm.sh all

# SDXL 使用同一份选择代码与启动脚本。
bash /home/shx/code/ce-new/casu-2/script/paper_su_middle.sh all --architecture sdxl
bash /home/shx/code/ce-new/casu-2/script/paper_su_vlm.sh all --architecture sdxl

# SD3 使用实际入口分组；按需开启扩散模型 CPU offload。
bash /home/shx/code/ce-new/casu-2/script/paper_su_vlm.sh all \
  --architecture sd3 --model-cpu-offload

# FLUX.2-klein 当前只有一个入口，显式允许 100% 的入口更新。
bash /home/shx/code/ce-new/casu-2/script/paper_su_vlm.sh all \
  --architecture flux --su-max-fraction 1

# 先只跑 Snoopy，或拆开擦除与正式采样阶段。
bash /home/shx/code/ce-new/casu-2/script/paper_su_vlm.sh erase --experiment instance_snoopy
bash /home/shx/code/ce-new/casu-2/script/paper_su_vlm.sh sample --experiment instance_snoopy

# 重复消融实验使用新的目录；erase 与 sample 指定相同 run-root。
bash /home/shx/code/ce-new/casu-2/script/paper_su_middle.sh all \
  --run-root /home/shx/code/ce-new/casu-2/runs/paper-sd1-su-middle-v2
```

这两组脚本默认模型为 SD1.4，支持 `--architecture sd1|sdxl|sd3|flux`，保存到 `runs/paper-<模型标签>-su-<middle或vlm>/`，例如 `runs/paper-sdxl-su-middle/`、`runs/paper-sd3-su-vlm/`；SD1 的原有新实验目录名不变。检查点在 `erase/<实验>/checkpoint/`，正式图片在 `images/<实验>/`，指标在 `metric/<run-root目录名>/<实验>/`。不会覆盖 `runs/paper-sd1/` 等旧实验。各架构继续复用自己的公共原图目录，相同采样配置无需重复生成。重复运行同一根目录时，擦除默认拒绝非空目录；采样默认 `--resume`，只复用匹配权重和配置的结果，权重不同会报错。保留旧结果请用新且名称不同的 `--run-root`，避免使用 `--overwrite`。

`--run-root` 不改变公共原图目录。如果改变 seed、采样步数、分辨率等原图配置，现有原图机制仍会覆盖不匹配的原图；若也要保留这些原图，请同时指定新的 `--original-dir /home/shx/code/ce-new/casu-2/runs/original_sd1_su_v2`。该参数只传给采样阶段。

擦除与 VLM 校准始终单 GPU 顺序执行，GPU 可在 `paper_su.sh` 的 `ERASE_DEVICE` 修改。完整数据采样沿用 `paper_sample_all.sh` 中的 `NUM_GPUS`、`DEVICE0`、`DEVICE1` 分配。脚本额外参数会分发到相应阶段，例如 `--su-vlm-model` 只传给擦除，`--num-samples` 只传给正式采样，`--seed` 同时传给两阶段。

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

SD3/FLUX.1 当前默认两个选择单元：序列入口与 pooled 入口。Klein 只有 context_embedder 一个选择单元，both 和 tokens 均编辑该入口；pooled 会报错。它们的统一 SU 当前在这些入口组之间选择，不能进行 block 级筛选；旧 legacy 也可逐入口撤去更新。它们均只接受文本条件，不修改图像输入、时间/guidance 分支或 single-stream 共享权重。图像生成仍会受到文本条件变化影响，不能据此保证图像内容完全不变。

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

四种架构统一使用薄 SVD 表示保留子空间，将目标输入投影到其正交补，并在 float64 下求解正定线性系统，不显式计算矩阵逆。相同输入空间的 K/V 复用求解，最终更新为 $W+\Delta W$，bias 保持原值。

求解设备默认跟随 `--device`，例如 `--device cuda:1` 会在 GPU 1 上求解。可用 `--solve-device cpu` 将求解单独放到 CPU，模型加载与生成设备不受影响。

求解器返回低秩因子 $L,R$，按样本数与输入维度选择较小的 Gram 系统，不构造完整零空间基；参数更新统一按 $(WL)R$ 计算。`solve_transform()` 保留为原闭式解的数值对照，不再用于运行时擦除。SDXL pooled 分支只将更新写入文本列，元数据列逐元素不变。

默认从本目录 instance.csv 或 style.csv 取前 128 个过滤去重后的非目标概念，保护其最后一个 subject token；另外保护完整空提示序列。可使用：

- --retain-count 0：不加入外部概念，空提示保护仍默认启用。
- --retain-count -1：使用全部保留概念，可能让保留空间满秩。
- --retain-path：替换 CSV 或逐行文本；CSV 列由 --retain-column 指定。
- --no-protect-null：关闭空提示保护，仅建议用于消融。
- --rank-rtol：数值秩容差，默认 1e-6。
- --preserve-rank：显式截断保护秩，会放松完整保留约束。

若保留空间满秩或目标没有可编辑方向，程序报错，不输出一个静默无效的“擦除成功”结论。metadata 中保存秩与数值约束诊断；这些不是图像效果指标。

### 统一选择器保留的 legacy 逐层删除流程

1. 求出固定的全部候选更新。
2. 全部启用，生成校准图片，判断初始组合。
3. 每轮仅撤去一个选择单元的更新，原网络结构不变。
4. 用相同提示、seed、采样配置重新生成。
5. 任一目标重新出现时，恢复本轮更新，停止并保存上一轮组合。

默认用一次原模型轨迹上的去噪响应代理排序，时间点覆盖整个采样过程。它只决定顺序，不作为擦除成功或停止判据；不使用旧 attention map 差值或固定 Top-K。也可选择 --selection-order forward、reverse 或 random。

SDXL/SD3/FLUX 的排序在原生 pipeline 的真实 backbone 输入上取样，匹配 latent/timestep，再更换文本条件计算参考响应。它保留原生条件字段、FlowMatch scheduler 和各代 FLUX 的 latent 布局；比较的是 backbone 预测输出，作为排序代理，不是最终图像判别。Klein 输入原生 patched 4D 噪声，由管线内部生成 4D 位置编码和打包；不会套用 FLUX.1 的 3D packed 输入。采集轨迹时使用 latent 输出，不解码图片。

固定顺序最多判断 n+1 个组合。首次失败即停止对顺序敏感，不声称得到全局最小模块集合。开启 SU 时同时保存 checkpoint_all，便于中断后仍可检查全更新模型。

## 4. 可选的多模态判别接口

旧逐层删除流程默认不调用接口。若已有本地多模态服务或兼容的视觉 chat API：

~~~bash
# 如果服务需要鉴权，凭据仍通过 VLM_API_KEY 提供，不写入运行配置。
bash script/erase.sh instance_snoopy \
  --selective-update --selection-method legacy --selection-judge vlm \
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

`sample.sh` 默认传入 `--save-comparisons`，将原始模型图片放在左侧、编辑模型图片放在右侧，保存到 `runs/paper-<架构>/images/<实验名>/<概念>/compare/`。两个 `paper_sample_all` 脚本也使用此设置。已有图片在配置一致时通过 `--resume` 复用并补存对照图；只调用 `main.py sample --benchmark` 时，需要显式添加 `--save-comparisons`。

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

公共原图按检查点架构分别保存在 `runs/original/`、`runs/original_SDXL/`、`runs/original_SD3/`、`runs/original_FLUX/`，每个根目录下按任务 / 概念保存。配置和提示相同时复用有效 PNG，改变采样配置时重新生成。实验目录中的 `original` 是指向公共原图任务目录的相对软链接，不复制图片；`manifest.json` 和 `index.html` 使用 `original/hello_kitty/000001.png` 这样的相对路径，编辑图片保存在本实验目录。

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
