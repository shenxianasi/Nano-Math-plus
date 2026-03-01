### Nano-Math-plus版本升级
- 1 针对原始Nano-Math无法做到对数学问题极长思维链的输出，我在Nano-Math这个SFT模型的基础上，进行了GRPO后训练，让模型能够对数学问题进行极长的思维链输出。同时，我也保留了模型对数学问题快速解决的能力，即：
用户可以不点击任何按钮，直接上传一张包含数学问题的图片，模型会用最短的篇幅（最长不会超过1024token）输出该问题的答案，如下图所示：
![alt text](figure/image.png)
如果用户需要对该问题进行更详细的解释，或者需要模型对该问题进行更复杂的计算，用户可以点击“深度思考”按钮，模型会输出更详细的解释（输出无最大上限），如下图所示：
![alt text](figure/image1.png)
- 2 本项目现采用流式输出，模型能够实时输出答案，用户无需等待模型处理完成，即可查看答案。
- 3 Nano-Math-plus模型也支持“联网搜索”功能，用户可以通过联网搜索扩展模型的先验知识，不仅能够扩展模型解题的能力，同时也能够让模型解决一些通用性问题，比如“今天北京的天气如何？”这样的问题。

### Nano-Math-plus项目文件结构及其功能
- 顶层概览
  - WebUI：Web 前端与后端服务，负责图片上传、模型推理、流式输出与联网搜索
  - tensorboard_export：训练阶段从 TensorBoard 导出的各类指标 CSV 与自动绘图脚本
  - data_process：数据处理脚本（样本抽样、转换、修复等）
  - dataset-verl：训练/验证/测试数据集（json、parquet）
  - config：训练相关配置
  - src：训练入口脚本
  - merged_safetensors / merged_hf / best_checkpoint / pretrained_model：模型权重与检查点
  - verl：强化学习训练框架与文档（集成自 VERL）
  - figure：示意图

- 关键目录与文件说明
  - WebUI
    - [app.py](WebUI/app.py)：Flask 后端
      - 加载 Qwen2.5-VL 推理模型，接收图片/文本输入
      - 普通/深度思考两种解题模式（动态 max_new_tokens）
      - 流式推理输出（TextIteratorStreamer）
      - 联网搜索（DuckDuckGo/Bing 回退）与天气直连查询（Open‑Meteo）
    - [templates/index.html](WebUI/templates/index.html)：前端页面
      - 图片上传、参数切换（深度思考/联网搜索）、消息流式渲染与滚动展示
    - WebUI/icons
      - generate_logo.html 等静态资源
  - tensorboard_export
    - 各类训练指标 CSV（如 .__actor__entropy.csv、.__perf__throughput.csv 等）
    - [plot_tensorboard_csvs.py](tensorboard_export/plot_tensorboard_csvs.py)：批量读取 CSV 并分页生成多子图拼图（plots_page_*.png）
    - plots_page_*.png：自动生成的指标总览图片（单页 24 个子图）
  - data_process
    - [10p_samples.py](data_process/10p_samples.py)：样本抽样
    - [data_transfer.py](data_process/data_transfer.py)：数据迁移/拷贝
    - [json2parquet.py](data_process/json2parquet.py)：JSON 转 Parquet
    - [repair_dataset.py](data_process/repair_dataset.py)：数据修复与清洗
  - dataset-verl
    - train/valid/test：训练/验证/测试集，含 data.json 与 data.parquet
  - config
    - [train_config.yaml](config/train_config.yaml)：GRPO 等训练配置
  - src
    - [run_grpo.sh](src/run_grpo.sh)：GRPO 训练脚本入口
  - 模型相关
    - best_checkpoint/[...]/：训练中间/最优检查点（如 global_step_650/data.pt）
    - merged_hf/[...]/：合并后的 HuggingFace 结构模型（含 tokenizer、config、chat_template 等）
    - merged_safetensors/[...]/：部署推理使用的最终权重与配置（被 WebUI 加载）
    - pretrained_model/[...]/：原始基础模型与分词器配置（SFT 基座）
  - verl（训练框架与文档）
    - docs/：算法与使用文档（GRPO/OP(O)/SPPO、FSDP、Ray、vLLM 等指南）
    - examples/：各类训练/推理示例脚本（如 grpo_trainer/*）
    - verl/：核心代码（trainer、workers、utils、single_controller 等）
    - docker/、.github/：构建与 CI 配置

- 典型流程
  - 数据准备：data_process → dataset-verl/{train,valid,test}
  - 训练与导出：src/run_grpo.sh + config/train_config.yaml → best_checkpoint → merged_hf/merged_safetensors
  - 可视化：训练同步到 TensorBoard → 导出到 tensorboard_export/*.csv → 运行 plot_tensorboard_csvs.py 生成拼图
  - 在线推理：WebUI 加载 merged_safetensors 模型，支持图片题目解答、深度思考与联网搜索（含天气直连）

### 环境配置
使用 veRL 框架进行 GRPO 后训练的环境依赖较多、变量配置复杂，容易产生冲突。这里提供一套可顺利启动训练的参考环境（建议 CUDA 12.4）：

```bash
# 创建与启用环境
conda create -n verl python=3.10 -y
conda activate verl

# 基础依赖
pip install vllm==0.8.2
pip install tensordict==0.6.2
# 可选：如需 sglang
# pip install sglang==0.4.5.post3
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install ray==2.44.0
```

从以下地址下载与环境匹配的 FlashAttention 预编译包（示例 v2.6.3），并在本地安装（若已下载到项目根目录，可直接使用）：

```
https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3
```

```bash
pip install xxx.whl
```

安装 veRL（推荐固定到 0.5.0 版本）：

```bash
git clone https://github.com/volcengine/verl
cd verl
git checkout v0.5.0
pip install -e .
```

注意：请先从 GitHub 安装好 0.5.0 版本的 veRL 框架，再结合本项目中的修改进行奖励函数及其他源码的调整。

### 针对数学题进行LCoT的GRPO后训练，自定义奖励函数
本项目在 VERL 的奖励机制上新增了一个面向数学题长链推理的自定义奖励管理器，并提供两种扩展方式：直接替换“评分函数”或新增“奖励管理器”。

- 功能概述（本项目内置）
  - 位置：`verl/verl/workers/reward_manager/my_reward_manager.py`
  - 目标：鼓励正确且规范的 LCoT 推理，同时抑制无意义的冗长输出
  - 评分构成（加权求和，区间裁剪到 [-1.5, 2.5]）
    - 正确性：基于 `prime_math.grader.math_equal` 判断等价，必要时回退精确匹配
    - 格式：优先抽取 `\boxed{...}` 中的最终答案，失败时回退到 GSM8K 提取器
    - 推理质量：依据逻辑指示词、数学术语、段落结构等启发式信号打分
    - 长度偏好：小于 `min_cot_length` 不奖励；`[min_cot_length, max_cot_length]` 线性增益；超过上限衰减
  - 形状与回填：计算出的“总分”只写入每条响应的“最后一个 token”位置（token-level reward）
  - 调试输出：`num_examine` 控制样例打印

- 接口说明（本项目内置）
  - 类注册名：`my_reward_manager`（通过 `@register("my_reward_manager")`）
  - 构造参数：`__init__(tokenizer, num_examine, compute_score=None, reward_fn_key="data_source")`
  - 调用签名：`__call__(data: DataProto, return_dict=False)`
    - 输入：`DataProto`
      - 文本：`batch["prompts"]`、`batch["responses"]`、`batch["attention_mask"]`
      - 标注：`non_tensor_batch["reward_model"]["ground_truth"]` 或回退 `non_tensor_batch["extra_info"]["answer"]`
    - 输出：
      - `return_dict=False` → `Tensor`（与 responses 等长，非末 token 为 0）
      - `return_dict=True` → `{"reward_tensor": Tensor, "reward_extra_info": Dict[str, List[float]]}`

- 在修改完`my_reward_manager.py`后，需要在`verl/verl/workers/reward_manager/__init__.py`中进行注册，添加如下代码：
```python
from .my_reward_manager import MyRewardManager
```
才能正常使用自定义奖励管理器。

### 针对VLM，对视觉输入进行预处理
首先判断输入的图片转换为视觉token之后是否超过2k tokens，如果超过进行等比例缩放，降低输入图片的分辨率，直到视觉token数量不超过2k tokens。而这一步主要是为了降低模型训练时的显存占用，避免因为视觉token数量过多而导致训练失败。

### 训练配置
我在autodl上租用了2张H800（80GB显存）进行训练，由于时间问题，我仅仅设置最大步数为700step，验证样本数为200个样本，训练时长大约在12~14个小时。
具体配置请参考`config/train_config.yaml`文件。

### 训练后将模型进行合并并导出为merged_safetensors
本项目使用 veRL 自带的 model_merger 将 FSDP 训练产物合并为 HuggingFace 结构的 safetensors 分片，然后复制到 `merged_safetensors` 目录供 WebUI 加载。以下是当时使用的合并与转化命令（相对路径）：

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir best_checkpoint/global_step_650/actor \
  --target_dir merged_hf/global_step_650
```

```bash
mkdir -p merged_safetensors
cp -r merged_hf/global_step_650 merged_safetensors/
```

### 训练日志输出
训练日志会保存在tensorboard_export文件下，我通过绘图工具，将所有输出信息都绘制出训练曲线进行可视化，比如：
![alt text](tensorboard_export\plots_page_2.png)
![alt text](tensorboard_export\plots_page_7.png)
![alt text](tensorboard_export\plots_page_29.png)

### 如果有哪些问题，欢迎在Issues中提出。
### 联系方式：邮箱18722164190@163.com
### 如果觉得本项目对你有帮助，欢迎给个Star⭐！