# 多场景控制频率压力实验

在项目要求的 CUDA 环境安装依赖、下载权重后，一条命令运行：

```bash
bash scripts/run_control_sweep.sh
```

默认读取 `web_client/scene_presets.yaml` 的全部 7 个场景；每个场景生成
`0.5, 1, 2, 3, 4, 5, 6, 8, 10 Hz` 的 A/D 往复方波视频，以及一段无振荡对照。
默认种子 42，每段 2 秒预热 + 12 秒刺激 + 2 秒恢复，共 **70 段、每段 16 秒**。
6/8/10 Hz 是明确标记的采样受限压力条件，不是有效的原频率响应估计。
输出位于 `outputs/control_frequency_sweep/<时间戳>/`；模型在一个进程中加载一次。

本工具实现**离线刺激回放与视频采集**。尚不自动计算 Bode 图、截止频率、
HF-Controllability Index 或几何涂抹分数，也不把视觉伪影直接判定为 latent manifold collapse。

## 常用命令

查看场景编号，或先在没有 GPU 的机器上验证计划（需要 `omegaconf`，不导入 PyTorch/CUDA）：

```bash
bash scripts/run_control_sweep.sh --list-scenes
bash scripts/run_control_sweep.sh --dry-run --output-dir outputs/control_frequency_sweep/plan
```

小规模真实推理：

```bash
CUDA_ID=0 bash scripts/run_control_sweep.sh \
  --scenes scene02 scene03 scene04 \
  --frequencies 0.5 1 2 4 \
  --duration 12 --annotate \
  --output-dir outputs/control_frequency_sweep/smoke
```

扩大实验矩阵（所有选项取笛卡尔积；注意视频总数）：

```bash
CUDA_ID=0 bash scripts/run_control_sweep.sh \
  --scenes all \
  --frequencies 0.5 1 2 3 4 5 6 8 10 \
  --waveforms square sine stop-go \
  --pairs AD JL WS \
  --seeds 42 43 44 \
  --duration 20 --warmup 2 --cooldown 2 \
  --annotate \
  --output-dir outputs/control_frequency_sweep/full
```

- `square`：第一键/第二键来回切换，例如 A→D→A→D。
- `sine`：`u(t)=a sin(2πft+φ)`，第一键为 `max(u,0)`，第二键为 `max(-u,0)`。
  **不会先转换成 bool**；分数幅值是针对二值键盘接口的实验性/OOD 条件。
- `stop-go`：第一键/全松交替；`--pairs WS --waveforms stop-go` 是 W→松开→W，
  只改变按键，不保证模型实现了物理意义的制动。
- `--pairs AD JL WS IK`：分别选择这四组相反方向键。正号总是该组第一个键，
  不预设光流或相机坐标的正方向。
- `--base-keys W --pairs AD`：持续前进时左右振荡；基底键不能与振荡键重叠或自身冲突。
- `--amplitude 0.5 --phase-deg 90`：固定幅值/起始相位；非 1 幅值也标为 OOD。
- `--no-baseline`：省略对照；默认对照只保持 `base-keys`，不会振荡。
- `--quant-type none --vae-type wan2.2`：切换量化/解码器以检查其对伪影的影响；
  所有推理设置写入记录。显存需求可能与默认配置不同。
- `PYTHON=/path/to/python bash scripts/run_control_sweep.sh ...`：指定已有 Python 环境。

自定义场景可复制现有 YAML，保留 `groups[].items[]`、`image`、`prompt`、`label` 结构，
通过 `--scenes-file /path/scenes.yaml` 指定。图片和 caption JSON 路径相对于该 YAML 的目录。
省略 prompt 继承 `default_prompt`，显式空字符串只使用原项目的 `| unknown |` 前缀。
参考图缓存按现有图片哈希规则查找；任何一张多视角参考图缺失时，沿用项目的零 reference mask，
并在每个实验记录中注明。

## 时间、采样与接口消融

频率单位中的秒采用仓库 `web_client/config.py` 的 **解码视频时间**（当前 12 FPS），
不是推理运行的墙钟时间，也不是前端真实按键事件率。修改 MP4 播放 FPS 无法证明模型原生带宽提高。
12 秒刺激对应 0.5 Hz 的 6 个周期；少于 4 周期会写入提示，建议增加 `--duration`。

| 控制模式 | 输入保持方式 | 当前名义采样率 | 基频需严格小于 |
| --- | --- | --- | --- |
| `--control-mode block` | 每生成块采样一次，块内保持（原始接口） | 1 Hz | 0.5 Hz |
| `--control-mode latent` | 每 latent 对应的 4 帧保持一个动作 | 3 Hz | 1.5 Hz |
| `--control-mode frame`（默认） | 每个视频帧一个动作，打包进 32 通道 | 12 Hz | 6 Hz |

第一次解码输出 `1 + 4 × (3−1) = 9` 帧，其后每块 12 帧。首帧动作复制四次，之后每四帧一组，
按 `channel = key_index × 4 + temporal_slot` 打包。CSV 明确保存首块和其后各块的时间对齐；
最后一块多生成的帧仅作内部 padding，不写入 MP4，但仍记录在动作文件中。
每块解码帧数若与预期不一致会报错，不会悄悄错开时间轴。

**逐帧通道语义仍需验证**：公开推理代码的 `set_act()` 会将每个键复制到四个相邻通道；
新增的 `set_act_sequence()` 以这四个通道为时间槽。测试验证恒定动作时与原方法逐元素相等，
但这并不证明非恒定情况下与训练数据的打包顺序相同。默认 `frame` 应作为实验性 adapter 输入，
与 `latent`/`block` 对照；在确认训练动作定义之前，不应把其伪影单独归因于模型动力学失稳。
整个 block 的动作已预先给定，这也不等价于在线推理能够以 12 Hz 接收并立即响应新操作。

默认 `--alias-policy allow` 生成所有请求条件；在/超过 Nyquist 的文件名含 `SAMPLING_LIMITED`，
元数据的 `fundamental_below_nyquist=false`，并记录 `folded_fundamental_hz`。
在 12 FPS 下，8 Hz 的正弦基频折叠到 4 Hz，10 Hz 折叠到 2 Hz；恰好 6 Hz 也无法辨识一般相位。
`--alias-policy skip` 会跳过这些视频但保存计划和轨迹，`--alias-policy error` 会拒绝整个计划。
该标志只描述**基频**；理想方波含高次谐波，基频合格不代表所有谐波都无混叠。
每周期少于 8 个采样点也会提示波形采样粗糙。

## 输出与恢复

```text
<run>/
  manifest.json                完整实验计划、场景/源代码哈希
  runtime.json                 真实推理环境与权重路径/大小/mtime
  results.json / summary.csv   条件状态与采样信息
  index.html                  本地视频预览索引
  scene02/frame/seed42/AD_square_1Hz_<hash>/
    video.mp4                 干净视频，保留精确分辨率和帧率
    annotated.mp4             可选：频率/按键/时间 HUD；不可代替干净视频算指标
    actions.csv               每帧 command/applied、阶段、采样源帧、是否写入视频
    actions.json              应用动作、打包顺序及各 block 的帧索引
    case.json                 该条件的完整设置和限制说明
    result.json               状态、实际帧数、耗时、MP4 大小及 SHA-256
```

CSV 的 `command` 是视频帧时间线上请求的信号，`applied` 是经过 frame/latent/block 采样保持后
交给 adapter 的信号。值保存在转换成 bfloat16 之前；正弦幅值实际会受模型 dtype 舍入影响。
基底按键以各键列为准，`u_command/u_applied` 仅表示振荡分量。
此 JSON 由新脚本逐帧消费；不要交给旧 `scripts/inference.py --action-json`，后者会重新按块抽样。

每个条件均重置 KV、cross-attention、模型级缓存与 VAE 状态；同一场景的静态条件复用。
条件准备完成后重新设定种子，覆盖初始噪声和去噪调度器内部噪声，用于配对比较；
不保证不同 CUDA/量化内核或环境之间逐位一致。编码后实际读取 MP4 核验帧数、FPS 和尺寸，
全部通过后才标记 `ok`。视频按块写入，不把完整长视频常驻 CPU/GPU 内存。

断点续跑时，使用**完全相同参数**和输出目录，加上 `--resume`，例如：

```bash
bash scripts/run_control_sweep.sh --output-dir outputs/control_frequency_sweep/run01
bash scripts/run_control_sweep.sh --output-dir outputs/control_frequency_sweep/run01 --resume
```

原计划、相关源代码、场景资源不同会拒绝续跑。已完成视频需通过大小与 SHA-256 校验才跳过；
损坏或未完成视频会重跑。首次生成时的 MP4 帧数/FPS/尺寸校验结果由文件哈希保护。
需要生成新条件时，还会核对已记录的 GPU/PyTorch/CUDA 和权重路径/大小/mtime，避免混用环境；
权重未做全量内容哈希，正式实验应额外固定并记录权重版本。
全部视频已完成的续跑可以只验证文件而不加载模型。中断返回 130，失败返回非零状态码。
单个条件失败默认记录并继续，`--fail-fast` 可立即停止；全局模型加载失败不会重复尝试几十次。

`--dry-run` 只生成计划和轨迹，没有视频；在同机同设置下移除 `--dry-run` 并加 `--resume`
即可执行。跨机器路径不同的情况下请建立新输出目录。

## 验证

```bash
python -m pytest tests/test_control_frequency_sweep.py -q
```

测试涵盖方波/正弦/启停、Nyquist 标签、首帧和 block 对齐、恒定动作与原接口一致、
多时间槽打包、真正 MP4 的编码/回读、恢复、损坏文件重跑和失败退出。
MP4 测试使用临时目录中的合成帧，只验证流水线，不代表已运行真实模型或验证其控制质量。
