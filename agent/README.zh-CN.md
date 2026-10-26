# IBP Studio Photo Agent 0.2.0

这是基于 ImageBatchProcessor v1.1.0（3edbc74）的社区改版，非原作者官方发行。入口 IBPPhotoAgent.exe 启动深色原生 Studio、后台本地 API 与 Qwen；重复启动复用后台服务。保留原生 Vulkan 渲染、手动调节、批处理与配方功能。

## 使用

打开 RAW／JPEG／PNG／TIFF，在 Agent 面板选择本地学生或 API 教师，输入命令并生成。Agent 审阅窗口显示实际归档的结果，可比较原图并选择“满意／不满意”。满意样本进入监督训练；不满意样本及修改标注的历史保留，当前版本不对不满意样本做负向 SFT。原生主画布后续手动编辑不改变已生成预览的标注；Web 工作区的“应用调整”可创建可标注的新手动修正样本。

工具栏“API 设置”分别配置教师和学生的 URL、模型名、密钥；教师默认预留 DeepSeek 兼容接口。远端使用 OpenAI Chat Completions 兼容协议。密钥以 Windows DPAPI 保存，不写入训练样本；留空不覆盖已有密钥，勾选清除才移除。使用远程角色会发送命令、照片测量、EXIF 和选中的论文片段，不发送原图像素。

工具栏“训练”打开数据、导出训练集、训练 Qwen／MLP、选择已完成的个人权重。选择空项“预置模型”可回退。个人权重选择目前在本次运行有效，重启恢复预置模型。Qwen 训练时释放预置模型内存，训练完成后恢复。

默认双击进入原生 Studio；IBPPhotoAgent.exe --web 进入浏览器工作区（含完整滑块、RAG、训练、TIFF 导出）。--no-browser 仅运行 API。关闭原生窗口保留本地服务；Web 工作区“退出”终止后台服务。原生 Agent 一次处理一张；原版批处理功能仍保留。

## 默认目录

- 系统“文档”文件夹中的 IBP Training：可迁移的训练数据。目录解析遵循 Windows 文档重定向。
- originals：按 SHA256 去重的原图副本。
- samples/<id>：sample.json、原图／结果预览、配方、历史标注。
- exports：all-records.jsonl、accepted.jsonl、qwen-sft.jsonl。
- weights：MLP 数组、JSON 权重、输入输出契约、评估报告；Qwen LoRA 适配器、tokenizer、基座版本、评估报告。
- models：外部模型的 Hugging Face 缓存；默认 0.6B 可训练基座随安装器提供。
- papers：用户导入 PDF；不会随公共安装器分发。
- %LOCALAPPDATA%/IBPPhotoAgent：会话、API 设置、原始作业与 RAG SQLite。

IBP_TRAINING_DATA 与 IBP_AGENT_DATA 可覆盖目录用于隔离验证。卸载不删除个人文档和 AppData 数据；不要手动把这两处放进公开安装包。

## 模型与训练

安装器预置 Qwen3-1.7B Q4_K_M（约 1.11 GB 十进制），由官方 Q8_0 再量化；来源、哈希、转换方式在 models/manifest.json。没有视觉编码器。CPU 执行模型，Vulkan 执行图像处理；建议 Windows x64、16 GB 内存、支持 Vulkan 的显卡驱动。实际内存与耗时取决于照片、上下文、训练长度和其他程序。

附带隔离的 CPython 3.12、PyTorch 2.9.0 CPU、Transformers 4.57.1、PEFT 0.17.1；完整依赖见 training/requirements-lock.txt。“准备 Qwen 环境”检查附带依赖，无需全局 Python。另附完整可训练的 Qwen3-0.6B 基座（约 1.52 GB），默认 LoRA 训练无需下载模型。GGUF 用于推理，training/base/Qwen3-0.6B 用于训练；二者不是同一份权重格式。该基座来自 Qwen 官方 ModelScope 仓库，版本和各文件 SHA256 在其 manifest.json 中。

CPU 默认仅开放 Qwen3-0.6B LoRA 训练；界面保留 1.7B／4B 选项供外部 CUDA 训练环境。附带的是 CPU PyTorch，不会自动切换显卡包。已在 i7-1165G7、16 GB、无 CUDA 的本机验证官方 0.6B 基座的离线 LoRA 更新、保存及重新加载推理。验证使用隔离的合成样本，只证明训练链路；完整个人数据训练的耗时与审美效果需另行评估，不能承诺交互速度。

Qwen 使用人工认可的参数 JSON 做 response-only SFT，可视作教师输出的任务蒸馏；不是复制教师整个网络，也不自动拥有教师全部能力。输入包括命令、结构化意图、原始统计和论文证据，目标是实际认可的参数。相机候选 MLP 读同一结构化数值子集，直接回归参数；不负责自由语言理解。MLP 的 weights.json / contract.json 可供后续移植，相机端必须复现测量口径并标定 ISP 参数转换。

训练至少要求 8 组互不重复的满意样本用于流程保护；这只是允许启动的下限，不是学会风格的样本量保证。同原图和同拍摄组不会跨训练／验证集合。Qwen 报告留出 token loss，MLP 报告参数 MAE 与训练中位数基线；两者均不能替代人工视觉评估。训练按钮不会凭空生成个人权重。

## 测量与保护

流程：论文检索 → 术语／保留条件 → 原图测量 → 教师或学生提出参数 → 渲染检查 → 人工标注。保存 23 项统计、256 桶 YRGB 直方图、全尺寸基线、可取得的 EXIF／RAW 白电平信息；模型输入使用压缩的 16 桶分布。候选仅按 1、0.75、0.5、0.25、0 缩小提出的调整，选择通过保护线的最大步幅。实际输出、原始提案和所有尝试分别归档。全尺寸 TIFF 导出再次检查。

论文用来核对术语和测量定义。各目标记录归一化尺度；工程目标／保护阈值不冒充论文审美标准。通过保护线不等于无失真；无法从全局统计推断主体、天空或肤色等语义区域，也不能恢复传感器截断和失焦丢失的真实信息。

RAG 为 SQLite FTS5/BM25＋中英文术语扩展、页级分块、查询缓存和增量索引。最多选 3 段、每段 500 字符进入模型。新安装可在“论文”中导入 PDF；文件名／PDF 标题不自动等于已核实的书目身份。公共安装器只携带术语表与书目元数据，不含用户 PDF 或全文数据库。

## 本地 API 与开发

服务仅绑定 127.0.0.1，每次随机 X-IBP-Token；session.json 供本机客户端取得地址和令牌。Host/Origin 检查保持开启。接口包括 /api/jobs、/api/manual、/api/feedback、/api/settings、/api/training、/api/weights/activate、/api/rag/search、/api/rag/import。设置与模型密钥不能写到示例命令或版本库。

标准 CPython 开发环境安装 agent/requirements-dev.txt。运行 agent/run_agent.py；测试 python -m unittest discover -s tests/agent -q；本地和安装版完整验证使用 tools/agent/verify_ui.py。native Studio 在原版构建中编译，需要 C++20、SDL、ImGui、Vulkan 开发环境与 external/json 中保留许可的头文件。

打包入口 tools/agent/build_windows.py，Inno 配置 packaging/agent.iss。包内 sources 保存 Agent、改动的 Studio 与依赖来源；保留所有第三方通知。安装器未签名。本社区 fork 的源码发布不等于安装器已公开发布，也不代表原作者认可此改版。
