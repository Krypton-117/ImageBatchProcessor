# 自然语言修图 → 可测指标 → 可控操作：文献与实现边界

核查日期：2026-10-26（Asia/Shanghai）。适用架构：本地纯文本 Qwen3-0.6B + llama.cpp → 结构化意图枚举 → 文献／规则检索 → 有界 GPU 候选评分。本文是研究依据，不是已实现功能或效果验证。稳定引文 ID 与机器可读元数据见同目录 `sources.json`；PDF 访问、许可和本地缺失原因也在其中。

## 1. 结论与证据层级

**可做的是解释用户的调整意图，在明确色彩域、作用区域和保护约束下比较有限候选；不能由几个图像统计量推导唯一正确的审美答案。** FiveK 的五位修图者和可接受范围研究支持多解与个人差异；它们并未证明某个亮度、对比度、彩度值适合所有照片。[fivek2011; acceptable2015]

本文标注三类内容，检索库也应保留该类型：

- **[D] 科学定义／论文测量模型**：说明测量对象和假设，未必说明应该往哪个方向调。[peli1990; hasler2003; foi2008; wang2004]
- **[P] 偏好／任务目标**：例如用户要低调、暖色、鲜艳，或训练数据代表某一群体的偏好；需要用户选择或任务上下文。[fivek2011; exposure2018; acceptable2015]
- **[H] 工程启发式**：本文提出的候选范围、百分位、权重、保护阈值和拒绝策略。均不是论文给出的普适美学标准；需要在自己的图像、输出设备和用户测试上校准。

`mitchell1997` 未绑定任何文献：本次没有确认其题名或 DOI。若原意是 1997 年多曝光 HDR 恢复，正确 ID 为 `debevec1997`，作者是 Paul E. Debevec 与 Jitendra Malik。不要生成作者误配的别名。

## 2. 先固定测量域

[D] 传感器 RAW、场景线性辐射／曝光、显示线性 RGB、带传递函数的 RGB 编码、显示端感知不是同一层。多曝光 HDR 恢复依赖响应函数与曝光信息；显示 tone mapping 解决的是另一问题。[debevec1997; reinhard2002]

[H] 每份 metric 记录至少包含：`metric_id, domain, primaries, white_point, transfer_function, range, bit_depth, resize, roi, alpha_policy, version, confidence`。评分前做一致的解码和色彩管理；线性 RGB 内做曝光乘法。固定 sRGB/D65 下可用线性相对亮度 `Y = 0.2126 R + 0.7152 G + 0.0722 B` 作为工程约定；换色域必须换系数。这里不是绝对 cd/m²，也不是从图像推导的场景照度。缺失 profile 时记录假设而不是冒充实测。

[H] 同图候选应使用相同预览尺寸、采样规则和 ROI；原分辨率复核剪切与光晕。透明区不计入直方图。RGB 编码值的色彩统计、线性 Y 的曝光统计、Lab 的色度统计分开命名。不要混合不同域和量纲形成“质量总分”。

## 3. 指标、操作和边界

| 指标与层级 | 定义／建议计算域 | 能支持的操作和证据 | 不可推出的结论 |
|---|---|---|---|
| 曝光与亮度 [D/H] | 线性曝光倍率 `2^e`，e 单位 stop；报告 Y 的中位数、分位数、对数均值，后几项是实现选取 | 线性曝光、色调曲线、阴影／高光分段调整；tone mapping 中的对数平均亮度有文献依据 [reinhard2002] | 像素均值低不等于拍摄曝光错误；夜景不必变成白天 |
| 预览端亮／暗端堆积 [H] | 浮点输出裁切前，分通道统计超出合法范围的比例；另在编码预览统计接近端点的比例，并注明阈值与量化范围 | 高光 roll-off、白点、黑点、曝光；观察面积和位置而非只看全图百分比 [debevec1997; foi2008] | 预览里的 255 可能是渲染白、反光、色域转换或压缩；不能证明 RAW 的传感器饱和，也不能证明底片可救 |
| 动态范围 [D/H] | 物理范围是有效最大／最小信号之比，以 stop 表示时取 log2；需规定噪声底、饱和和测量方式。`log2((P99(Y)+eps)/(P1(Y)+eps))` 仅是本文预览百分位跨度代理 | tone mapping 压缩显示跨度；有原始多曝光数据时才考虑辐射恢复 [debevec1997; reinhard2002] | 不能把 8-bit 的 256 级直接叫 8 stop 场景动态范围；eps 主导的黑图结果不可信 |
| 噪声 [D] | RAW 的 Poissonian–Gaussian 模型用 `Var(z|y)=a*y+b` 表达信号相关和近似信号无关项；参数需估计，裁切会影响统计 [foi2008] | 匹配噪声模型的去噪、曝光与阴影增益限制；另记录纹理损失 | JPEG 高频、压缩方块、头发与草地不能直接作为传感器噪声测量；参数不能跨 ISO／管线无条件复用 |
| 全局／局部对比 [D/H] | Michelson `(Lmax-Lmin)/(Lmax+Lmin)` 适合受控简单图案；一般照片可另报 Y 标准差、分位跨度，但它们只是全局代理。Peli 定义按空间频带的局部对比：带通响应／更低频局部均值 [peli1990] | 曲线、全局对比、局部对比；记录窗口、尺度、边界处理；使用边缘保护 [peli1990; reinhard2002] | 标准差增加不保证感知更清楚；局部对比会放大噪声或产生光晕 |
| 色彩丰富度 [D] | Hasler–Süsstrunk 的 RGB 对手色近似：`rg=R-G`, `yb=(R+G)/2-B`; `C=sqrt(std(rg)^2+std(yb)^2)+0.3*sqrt(mean(rg)^2+mean(yb)^2)`。RGB 数值范围需固定，不能直接比较 0–1 与 0–255 结果 [hasler2003] | 用同一实现比较彩度操作前后趋势，并加通道剪切／越色域检查 | 色彩丰富度是感知属性估计，不是审美分、准确色还原分或白平衡分；灰阶作品的低值可以完全符合意图 |
| 白平衡／色偏 [H] | 对已知中性参考区域统计色度偏移；没有可靠参考时只能给候选，不断言光源色温。全图均值中性是附加假设 | 白平衡增益、色温／色调 UI、局部颜色操作 [exposure2018] | “偏黄”可能是暖光风格；JPEG 加暖不等于恢复物理照明或 RAW 白平衡 |
| SSIM [D] | 成对、对齐图像，局部窗口的亮度／对比／结构相似性；常用合并式 `(2μxμy+C1)(2σxy+C2)/((μx²+μy²+C1)(σx²+σy²+C2))`；动态范围、窗口、通道及聚合需固定 [wang2004] | 作为参考图保真或未编辑区保护的一个量；不应唯一决定修图成败 | 直接最大化输入与输出 SSIM 会偏向不改图；对故意改曝光、风格会惩罚合理修改 |
| LPIPS [D] | 学习到的深层特征距离，需规定网络版本、归一化、分辨率与参考；源工作检验与人类感知判断的相关性 [zhang2018] | 可选的参考差异量；它需要额外视觉网络，不属于纯手工指标，也不等于语义 VLM | 不是无参考美学分，也不能证明人物身份／内容保持；当前严格指标路径可以不启用 |
| 局部约束 [H] | 对用户提供掩膜内外分别算指标、边缘梯度及修改幅度；报告蒙版来源与覆盖率 | 区域曝光、饱和度、对比度。GIER 将操作与区域关联，说明局部 grounding 是额外任务 [gier2021] | 纯文本解析器加全图统计没有区域定位能力，不应猜测天空、皮肤或主体 |

[D] Hasler 公式的来源是自然图像主观色彩丰富度实验；不同 RGB 编码、缩放、裁切和场景组成都会改变输入统计。本文不指定“最佳 C”。Reinhard 常见 key 参数 0.18 也不是所有照片必须达到的平均亮度目标。[hasler2003; reinhard2002]

## 4. 中文日常表达的保守映射

以下枚举与响应流程均为 [H] 实现建议；论文仅支持操作类型、可测概念或局限，不证明这些中文词有唯一含义。[exposure2018; planning2021; gier2021]

| 用户说法 | 建议结构化意图 | 优先测量／候选 | 需要明确的歧义与保护 |
|---|---|---|---|
| “太暗了，亮一点” | `BRIGHTEN`, scope=`GLOBAL` | Y 中位数和分位数；小幅曝光或曲线 | 保护亮端；夜景保持低调；不以填满直方图为目标 |
| “亮的地方白成一片，压一压” | `REDUCE_HIGHLIGHTS` | 亮端区域统计；高光肩部、曝光向下候选 | 只能压暗现有预览；没有源信息时不承诺恢复纹理 |
| “暗处看不清，但别让灯更亮” | `LIFT_SHADOWS`, `PRESERVE_HIGHLIGHTS` | 暗部曲线、低亮度范围权重；噪声代理与亮端保护 | 亮度范围掩膜不是物体语义掩膜；低光噪声可能更明显 |
| “灰蒙蒙，通透一点” | `AMBIGUOUS_CONTRAST_OR_HAZE` | 给曲线／黑点／局部对比备选 | 低对比、雾、逆光、镜头炫光不是一回事；不自动声称物理去雾 |
| “太硬了，柔和一点” | `REDUCE_CONTRAST` 或 `REDUCE_LOCAL_CONTRAST` | 全局跨度与局部频带指标 | “柔和”也可能指光照、肤质或模糊；先小幅候选 |
| “颜色淡，鲜艳一点” | `INCREASE_COLORFULNESS` | 彩度／饱和度小幅增加，比较 C 和通道风险 | 不假定彩度越高越好；肤色保护需要区域依据 |
| “颜色太假，别这么艳” | `DECREASE_COLORFULNESS` | 彩度向下，保留亮度层次 | 也可能是白平衡／局部色相问题；可返回多个解释 |
| “有点黄，冷一点” | `COOLER_COLOR_BALANCE` | 色彩增益或温度候选，已知中性 ROI 可校验 | 无参考时是用户偏好调整，不是测出了正确色温 |
| “暖一点，有黄昏感” | `WARMER_COLOR_BALANCE`, style=`USER_PREFERENCE` | 暖色小幅候选，维持通道余量 | 不假装改变真实照明或重建阴影 |
| “颗粒太多，干净一点” | `DENOISE` | 合适域中的噪声估计、细节保护 | JPEG 残差仅作代理；去噪与纹理保存冲突 |
| “不够清楚，清晰一点” | `AMBIGUOUS_SHARPNESS_OR_CONTRAST` | 局部对比、锐化的轻量备选 | 失焦、运动模糊、噪声、分辨率不足不同；不承诺恢复未记录细节 |
| “只把脸提亮，背景别动” | `BRIGHTEN`, scope=`MASK_REQUIRED` | 只有用户选区／可靠外部 mask 到位后执行 | 没有 mask 返回 `NEEDS_REGION`，不可退化成偷偷全图提亮 |
| “天空别过曝，人还要亮” | 两个区域意图，`MASK_REQUIRED` | 分区约束；可能无可行解 | 单张已渲染图片无法保证兼得，提供妥协结果并解释 |
| “自然一点／高级一点” | `AMBIGUOUS_STYLE` | 请求参照或展示几个弱候选 | 没有科学上通用的自然／高级目标 |

## 5. 文献对架构的支持与不支持

[D] Exposure 使用可解释、可微分的传统调整操作，并用图像条件策略和训练数据风格指导序列；它支持“参数化操作能形成可解释编辑”的主张，不证明纯文本小模型和统计评分能复现其结果。[exposure2018]

[D] Learning by Planning 将语言请求映射到操作序列，并用目标图规划训练监督；它支持文本到操作的设计参照，但其学习系统还利用图像信息，不能视为零训练、无图像输入的规则引擎验证。[planning2021]

[D] GIER 提供全局／局部语言编辑及操作、区域关联。它说明语义定位、操作选择和参数估计是可分解子任务；不代表纯文本模型本身能定位图中对象。[gier2021]

[H] 建议把上述成果用于检索规则的出处，禁止把论文的用户研究结果写成本项目的效果分数。当前不研究或声称 Qwen3-0.6B 的具体延迟、JSON 成功率、llama.cpp 性能或 GPU 吞吐；这些须在目标机器单独实测。

### 5.1 建议的意图契约

[H] 解析输出只包含固定枚举与受限字段，例如：

```json
{
  "intent": "LIFT_SHADOWS",
  "scope": "GLOBAL_LUMINANCE_RANGE",
  "strength": "LOW",
  "constraints": ["PRESERVE_HIGHLIGHTS"],
  "ambiguities": [],
  "needs_region": false
}
```

[H] 模型不要直接输出任意着色器、公式、文件路径或未检查的浮点参数。解析结果经过 schema 和枚举白名单验证；失败用确定性短语规则或澄清。否定（“不要更艳”）、程度（“一点”）、区域和矛盾约束要单独保留。引用与指标公式来自受控检索库，不能由模型补造。

[H] 检索以 `intent + metric domain + source availability + scope` 为键；规则需有 `citation_ids`, `evidence_type`, `preconditions`, `limitations`, `allowed_operators`。参考图保真、用户偏好、物理测量分开存储。`hasler2003` 的规则只能说彩度指标怎么测，不能把“C<某值就是不好看”伪装成该论文结论。

### 5.2 有界候选与评分

[H] 以下数字仅为最初可复现配置，不是科学阈值或已验证默认值：一次最多 32 个候选；每候选最多 2 个操作；预览长边 512 像素；曝光先尝试 `{-0.5,-0.25,0,+0.25,+0.5}` stop，按意图限制方向；饱和度乘数先尝试 `{0.9,1.0,1.1}`。保留不修改候选，并以任务配置控制上界。多操作顺序会改变结果，必须把顺序计入候选身份。

[H] 先过滤不合法输出、无掩膜的局部任务、非有限像素和明显违反保护约束的候选，再比较意图收益、输入变化与副作用。所有分量先固定域和尺度；不要直接加总 stop、像素比例和 SSIM。示意：

`score = w_intent * bounded_target_progress - w_clip * new_clip_risk - w_noise * noise_proxy_increase - w_change * edit_magnitude - w_outside * outside_mask_change`

[H] `bounded_target_progress` 应在用户要求的有限变化范围内饱和；不是无限提高彩度或亮度。`new_clip_risk` 比较输入与候选的新增风险，并同时保留绝对值；已有纯白背景不必判失败。对于用户明确要剪影、黑白或高调的情况，保护权重和目标必须随意图改变。不要用统一禁止黑白端点的规则抹掉风格。

[H] 噪声项只在估计器有可信平坦区域或适用 RAW 域时启用；否则记录 `unavailable`，不是填 0。无参考图时不计算“对真实正确结果”的 SSIM／LPIPS；输入参考只作为变化保护，不能独占总分。指标冲突时保留 Pareto 候选供用户选，不隐藏为一个权重未经验证的绝对质量排序。

[H] 对前几名候选做原分辨率端点／光晕复核，因为缩小图会隐藏剪切、细节损失和局部瑕疵。若目标不可行，返回 `NO_SAFE_CANDIDATE` 并解释约束，不越界扩大参数搜索。保存候选参数、指标前后值、域、依据 ID、失败原因；不要保存虚构的“人眼满意概率”。

## 6. 多方分歧与评价设计

| 分歧 | 原始证据支持什么 | 对实现的要求 |
|---|---|---|
| 五位修图者 vs 单一标准答案 | FiveK 专门研究个体偏好与差异学习；单个修图者可作某一任务参考 [fivek2011] | 分专家报告，不能把 expert C 当作普适真值；同一原图跨专家变体应在同一数据划分，避免泄漏 [H] |
| 多个可接受结果 vs 唯一最优 | 可接受范围研究收集亮度／对比度调整的判断，目标是辨别可接受范围 [acceptable2015] | 用户可以偏好多个版本；不能外推到所有文化、风格、输出设备 [H] |
| 全局统计 vs 感知局部对比 | Peli 强调位置与空间频率；单个全图数值可能漏掉视觉差异 [peli1990] | 同时报 ROI／多尺度，检查光晕；不宣称论文否定所有全局对比用途 [H] |
| 参考保真 vs 意图完成 | SSIM／LPIPS 的参考差异用途与有意改色、改曝光的目标不同 [wang2004; zhang2018] | 相似度下降可能是正确执行；另评指令满足与副作用 [H] |
| 无配对风格学习 vs 可解释指标搜索 | Exposure 学习数据风格；本文评分是手工指定的有限目标 [exposure2018] | 不借用 Exposure 的美学结果证明本实现；分别实测 [H] |
| 局部语义编辑 vs 纯统计系统 | GIER 的局部操作需要区域 grounding [gier2021] | 用户 mask 或另接定位模块；当前不能据指标虚构语义理解 [H] |

[H] 建议评价分三条线：① 中文否定、歧义和区域请求的解析正确率；② 固定图像上指标计算与操作边界的可复现性；③ 盲测用户的指令满足、可接受性与偏好，多人评分并保留分歧。报告分布和失败类型，不只报告平均“美感提升”。包括夜景、雪景、剪影、单色、皮肤、噪声暗部、JPEG 与 RAW 对照等反例；这些是建议测试设计，不是已经执行的实验。

## 7. 来源使用与访问清单

下列 12 篇核心来源的完整作者、题名、年份、刊会、卷期／页码、DOI、原始站点与 PDF 地址见 `sources.json`。公开可访问、开放许可、个人研究复制许可和可重新分发是不同状态。arXiv 的 non-exclusive distribution 许可授予 arXiv 分发权，不自动授予本项目再次分发权；代码许可证和 FiveK 图像协议也不等于论文 PDF 许可证。

- `debevec1997`：场景辐射与曝光恢复的边界；SIGGRAPH 1997，369–378，DOI `10.1145/258734.258884`。
- `reinhard2002`：摄影式 tone mapping；TOG 21(3)，267–276，DOI `10.1145/566654.566575`。大学仓库技术报告不是带出版排版的最终版。
- `peli1990`：局部、频带相关对比；JOSA A 7(10)，2032–2040，DOI `10.1364/JOSAA.7.002032`。
- `hasler2003`：自然图像彩度指标；SPIE 5007，87–95，DOI `10.1117/12.477378`。Crossref 简略记录仅列起页 87；终页应以完整出版记录／全文复核，清单保留此差异。
- `foi2008`：RAW 噪声建模；IEEE TIP 17(10)，1737–1754，DOI `10.1109/TIP.2008.2001399`。
- `wang2004`：SSIM；IEEE TIP 13(4)，600–612，DOI `10.1109/TIP.2003.819861`。
- `zhang2018`：LPIPS；CVPR 2018，586–595，DOI `10.1109/CVPR.2018.00068`。
- `fivek2011`：配对照片与个体偏好；CVPR 2011，97–104，DOI `10.1109/CVPR.2011.5995413`。
- `acceptable2015`：可接受亮度／对比度范围；ICCP 2015，1–9，DOI `10.1109/ICCPHOT.2015.7168372`。
- `exposure2018`：可解释操作序列；TOG 37(2)，Article 26，1–17，DOI `10.1145/3181974`；arXiv v1 是 2017 年，不能混同发表年份。
- `gier2021`：全局／局部语言编辑；*Computer Vision – ACCV 2020*，LNCS，636–651，DOI `10.1007/978-3-030-69544-6_38`。会议为 2020，书章出版为 2021，arXiv `2010.02330`。
- `planning2021`：语言到可解释全局操作；CVPR 2021，CVF 页码 13590–13599，DOI `10.1109/CVPR46437.2021.01338`。Crossref 登记 13585–13594 且作者 Trung 被拼成 Tmng；采用 CVF 原始页面的作者及页码，保留登记差异。

所有模型／算法原论文的适用假设与本项目工程推论已区分。未证实的下载状态不写作“已获取”；网页返回 200 但内容是访问挑战页面也不算获取论文。

## 8. CIE 色彩术语：不要把三个概念混用

补充的 `cie2020` 是 CIE S 017:2020《International Lighting Vocabulary》第 2 版，DOI `10.25039/S017.2020`，不是第 13 篇研究论文。以 CIE 官方 e-ILV 的下列条目为定义依据：

- **colourfulness／色彩丰富度、彩色感程度**（17-22-072）：感知到的颜色有多少彩色成分。Hasler 指标估计这一属性，不等于以下两个相对量。
- **saturation／饱和度**（17-22-073）：相对于该区域自身亮度感（brightness）的色彩丰富度。HSV/HSL 的 S 是特定颜色模型坐标，不自动等同于 CIE 的感知定义。
- **chroma／彩度**（17-22-074）：相对于在相似照明下看起来为白色或高透射参考区域的 brightness 的色彩丰富度。Lab 的 `C*ab=sqrt(a*²+b*²)` 是特定色空间中的度量，不是 Hasler C 的别名。
- **brightness／亮度感**（17-22-059）是感知属性；**lightness／明度**（17-22-063）涉及相对于相似照明下白色参考的亮度感。不要把它们与辐射计／光度计测量值、RGB 编码值、某个直方图均值混为一谈。

以上是术语释义，不是暖色、冷色、自然肤色或漂亮程度的目标值。CIE 术语也不支持“平均 Lab b 就是色温”的说法。`cie2020.term_urls` 记录官方逐条入口，`pdf_url=null` 表示未确认免费完整标准 PDF；只提供官方在线定义及标准书目，不下载付费标准。[cie2020]

## 9. `agent/knowledge/terms.json` 引文对应核查

只读核查快照：2026-10-26。共 13 个 terms；这里只审核定义／出处对应关系，未测试算子、度量实现、文本模型或 GPU 评分。主实现持续更新后需重新运行对应检查。[H]

| term ID | 当前来源 | 核查结论／建议 |
|---|---|---|
| `brighten` | reinhard2002, foi2008 | 部分直接支持：tone mapping 与 RAW 噪声分别对应操作、风险。exposure=0.4 与 shadows=0.15 是工程种子，文献不提供该默认值。 |
| `darken` | reinhard2002 | 色调调整有依据；关于源信息丢失的边界补 debevec1997，明确“已剪切”指输入数据中确实不可用，而非由预览判定 RAW。 |
| `highlights` | reinhard2002 | 支持显示范围压缩，不能据预览断定传感器饱和；补 debevec1997 作恢复条件出处。`天空太白` 若涉及区域，另需用户 mask。 |
| `shadows` | foi2008, reinhard2002 | 支持噪声与调整的风险；`noiseEstimate` 不是 Foi 原始传感器估计器除非实际实现其假设与方法。`脸太黑` 的全局阴影替代需明确告知，不能声称已定位人脸。 |
| `contrast` | reinhard2002, exposure2018 | 支持可控操作；建议加 peli1990 支持复杂图像局部／多尺度对比。`rmsContrast`、`edgeEnergy` 本身不能命名为 Peli 对比。 |
| `soft` | reinhard2002 | 可支持 tone 操作，CIE／Peli 可解释对比相关概念；并不支持光学柔焦或重打光能力。 |
| `warm` | hasler2003 | **出处不匹配**：Hasler 衡量彩度，不证明白平衡正确性或 Lab b 冷暖方向。建议以 exposure2018 支持白平衡操作，cie2020 支持术语；Lab b 方向代理保留 [H]。 |
| `cool` | hasler2003 | **出处不匹配**，同 warm。不能因为论文涉及颜色就把它当作所有色彩操作的依据。 |
| `vivid` | hasler2003, fivek2011 | 支持彩度指标及个体偏好；建议加 cie2020 区分 colourfulness 与 saturation；FiveK 不直接支持 saturation=1.12 或 vibrance=0.08。 |
| `muted` | hasler2003, fivek2011 | 同上，去饱和方向来自用户偏好；“高级灰”不是论文定义；建议加 cie2020。 |
| `denoise` | foi2008 | 支持 RAW 噪声模型；**不直接支持当前双边滤波实现或空间噪声代理有效性**。双边降噪、seed=0.2 与细节保护需标工程实现；不能宣称复现 Foi。 |
| `sharpen` | wang2004 | **出处不匹配**：SSIM 是参考质量／相似度度量，不是锐化算子或边缘能量定义。当前可将锐化标为 [H]，仅在实际使用 SSIM 保护时关联 wang2004；本研究未增补锐化算法原始文献。 |
| `natural` | fivek2011, exposure2018 | 支持风格／个人偏好与可解释处理；不能证明保守变化量就是普适“自然”审美标准。 |

[H] ID 存在只是第一层验证。`sources` 建议进一步拆成“术语定义出处”“测量定义出处”“操作出处”“风险出处”，或至少在可展示说明中指出各篇支持哪句话。当前文件的 `target_basis` 已将数值种子与审美标准分开，这个区分应保留。所有当前 source ID 都应在 manifest 中可解析；`peli1990` 与 `cie2020` 已可用于集成。`mitchell1997` 无已确认论文，不应加入知识库。


## 10. 完整书目与原始入口

以下链接均为出版社、会议、作者或机构仓库入口。PDF 为最终发表版还是作者稿分别说明；访问成功不等于开放再分发许可。

### [debevec1997]

Paul E. Debevec; Jitendra Malik. **Recovering High Dynamic Range Radiance Maps from Photographs**. 1997. SIGGRAPH 1997, pp. 369-378。

- DOI: [10.1145/258734.258884](https://doi.org/10.1145/258734.258884)
- 官方／作者入口：[https://www.pauldebevec.com/Research/HDR/](https://www.pauldebevec.com/Research/HDR/)
- 官方／作者 PDF：[https://people.eecs.berkeley.edu/~malik/papers/debevec-malik97.pdf](https://people.eecs.berkeley.edu/~malik/papers/debevec-malik97.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Primary source catalogued; local PDF retrieval failed: curl: (28) Operation timed out after 150009 milliseconds with 674489 out of 1434221 bytes received
- 本地副本：无；原因见下表。

### [reinhard2002]

Erik Reinhard; Michael Stark; Peter Shirley; James Ferwerda. **Photographic Tone Reproduction for Digital Images**. 2002. ACM Transactions on Graphics 21(3), pp. 267-276。

- DOI: [10.1145/566654.566575](https://doi.org/10.1145/566654.566575)
- 官方／作者入口：[https://doi.org/10.1145/566654.566575](https://doi.org/10.1145/566654.566575)
- 官方／作者 PDF：[https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf](https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Public primary-source PDF downloaded and parsed; research-only local copy
- 本地副本：`docs/research/papers/reinhard2002.pdf`；SHA-256 见 manifest。
- 版本／书目备注：PDF is university technical report UUCS-02-001, dated 2002-01-14, not final TOG typesetting.

### [peli1990]

Eli Peli. **Contrast in complex images**. 1990. Journal of the Optical Society of America A 7(10), pp. 2032-2040。

- DOI: [10.1364/JOSAA.7.002032](https://doi.org/10.1364/JOSAA.7.002032)
- 官方／作者入口：[https://opg.optica.org/josaa/abstract.cfm?uri=josaa-7-10-2032](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-7-10-2032)
- 官方／作者 PDF：[https://pelilab.partners.org/papers/ContrastJOSA.pdf](https://pelilab.partners.org/papers/ContrastJOSA.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Primary source catalogued; local PDF retrieval failed: curl: (28) Failed to connect to pelilab.partners.org:443 after 22967 ms: Could not connect to server
- 本地副本：无；原因见下表。

### [hasler2003]

David Hasler; Sabine E. Süsstrunk. **Measuring colorfulness in natural images**. 2003. Human Vision and Electronic Imaging VIII, Proceedings of SPIE 5007, pp. 87-95。

- DOI: [10.1117/12.477378](https://doi.org/10.1117/12.477378)
- 官方／作者入口：[https://infoscience.epfl.ch/entities/publication/e64ec735-2e1b-45ea-bc83-71f6d735068e](https://infoscience.epfl.ch/entities/publication/e64ec735-2e1b-45ea-bc83-71f6d735068e)
- 官方／作者 PDF：[https://infoscience.epfl.ch/server/api/core/bitstreams/77f5adab-e825-4995-92db-c9ff4cd8bf5a/content](https://infoscience.epfl.ch/server/api/core/bitstreams/77f5adab-e825-4995-92db-c9ff4cd8bf5a/content)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Official EPFL open-access repository PDF downloaded and verified
- 本地副本：`docs/research/papers/hasler2003.pdf`；SHA-256 见 manifest。
- 版本／书目备注：Formula verified in Section 7 of author manuscript. Publisher spelling colorfulness differs from manuscript colourfulness. Crossref only lists start page 87.

### [foi2008]

Alessandro Foi; Mejdi Trimeche; Vladimir Katkovnik; Karen Egiazarian. **Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data**. 2008. IEEE Transactions on Image Processing 17(10), pp. 1737-1754。

- DOI: [10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)
- 官方／作者入口：[https://webpages.tuni.fi/foi/sensornoise.html](https://webpages.tuni.fi/foi/sensornoise.html)
- 官方／作者 PDF：[https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Primary source catalogued; local PDF retrieval failed: curl: (28) Operation timed out after 150006 milliseconds with 3468072 out of 6751576 bytes received
- 本地副本：无；原因见下表。

### [wang2004]

Zhou Wang; Alan C. Bovik; Hamid R. Sheikh; Eero P. Simoncelli. **Image Quality Assessment: From Error Visibility to Structural Similarity**. 2004. IEEE Transactions on Image Processing 13(4), pp. 600-612。

- DOI: [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)
- 官方／作者入口：[https://www.cns.nyu.edu/~lcv/ssim/](https://www.cns.nyu.edu/~lcv/ssim/)
- 官方／作者 PDF：[https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf](https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Official/author public PDF downloaded and parsed for local research
- 本地副本：`docs/research/papers/wang2004.pdf`；SHA-256 见 manifest。

### [zhang2018]

Richard Zhang; Phillip Isola; Alexei A. Efros; Eli Shechtman; Oliver Wang. **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric**. 2018. CVPR 2018, pp. 586-595。

- DOI: [10.1109/CVPR.2018.00068](https://doi.org/10.1109/CVPR.2018.00068)
- 官方／作者入口：[https://richzhang.github.io/PerceptualSimilarity/](https://richzhang.github.io/PerceptualSimilarity/)
- 官方／作者 PDF：[https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Public primary-source PDF downloaded and parsed; research-only local copy
- 本地副本：`docs/research/papers/zhang2018.pdf`；SHA-256 见 manifest。

### [fivek2011]

Vladimir Bychkovsky; Sylvain Paris; Eric Chan; Frédo Durand. **Learning Photographic Global Tonal Adjustment with a Database of Input/Output Image Pairs**. 2011. CVPR 2011, pp. 97-104。

- DOI: [10.1109/CVPR.2011.5995413](https://doi.org/10.1109/CVPR.2011.5995413)
- 官方／作者入口：[https://people.csail.mit.edu/vladb/photoadjust/](https://people.csail.mit.edu/vladb/photoadjust/)
- 官方／作者 PDF：[https://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf](https://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Primary source catalogued; local PDF retrieval failed: curl: (18) end of response with 1519988 bytes missing
- 本地副本：无；原因见下表。
- 版本／书目备注：Dataset licenses are separate from paper copyright.

### [acceptable2015]

Ronnachai Jaroensri; Sylvain Paris; Aaron Hertzmann; Vladimir Bychkovsky; Frédo Durand. **Predicting Range of Acceptable Photographic Tonal Adjustments**. 2015. ICCP 2015, pp. 1-9。

- DOI: [10.1109/ICCPHOT.2015.7168372](https://doi.org/10.1109/ICCPHOT.2015.7168372)
- 官方／作者入口：[https://projects.csail.mit.edu/acceptable-adj/](https://projects.csail.mit.edu/acceptable-adj/)
- 官方／作者 PDF：[https://projects.csail.mit.edu/acceptable-adj/acceptable_adj.pdf](https://projects.csail.mit.edu/acceptable-adj/acceptable_adj.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Public primary-source PDF downloaded and parsed; research-only local copy
- 本地副本：`docs/research/papers/acceptable2015.pdf`；SHA-256 见 manifest。

### [exposure2018]

Yuanming Hu; Hao He; Chenxi Xu; Baoyuan Wang; Stephen Lin. **Exposure: A White-Box Photo Post-Processing Framework**. 2018. ACM Transactions on Graphics 37(2), Article 26, pp. 1-17。

- DOI: [10.1145/3181974](https://doi.org/10.1145/3181974)
- 官方／作者入口：[https://arxiv.org/abs/1709.09602](https://arxiv.org/abs/1709.09602)
- 官方／作者 PDF：[https://arxiv.org/pdf/1709.09602](https://arxiv.org/pdf/1709.09602)
- 许可／访问：arXiv non-exclusive distribution license. Local research only; exclude PDF from installer. Primary source catalogued; local PDF retrieval failed: curl: (28) Operation timed out after 150012 milliseconds with 1504423 out of 6579154 bytes received
- 本地副本：无；原因见下表。

### [gier2021]

Jing Shi; Ning Xu; Trung Bui; Franck Dernoncourt; Zheng Wen; Chenliang Xu. **A Benchmark and Baseline for Language-Driven Image Editing**. 2021. Computer Vision - ACCV 2020, LNCS, pp. 636-651。

- DOI: [10.1007/978-3-030-69544-6_38](https://doi.org/10.1007/978-3-030-69544-6_38)
- 官方／作者入口：[https://arxiv.org/abs/2010.02330](https://arxiv.org/abs/2010.02330)
- 官方／作者 PDF：[https://arxiv.org/pdf/2010.02330](https://arxiv.org/pdf/2010.02330)
- 许可／访问：arXiv non-exclusive distribution license. Local research only; exclude PDF from installer. Primary source catalogued; local PDF retrieval failed: curl: (28) Operation timed out after 150014 milliseconds with 2925120 out of 34841158 bytes received
- 本地副本：无；原因见下表。
- 版本／书目备注：ACCV conference 2020; proceedings chapter published 2021.

### [planning2021]

Jing Shi; Ning Xu; Yihang Xu; Trung Bui; Franck Dernoncourt; Chenliang Xu. **Learning by Planning: Language-Guided Global Image Editing**. 2021. CVPR 2021, pp. 13590-13599。

- DOI: [10.1109/CVPR46437.2021.01338](https://doi.org/10.1109/CVPR46437.2021.01338)
- 官方／作者入口：[https://openaccess.thecvf.com/content/CVPR2021/html/Shi_Learning_by_Planning_Language-Guided_Global_Image_Editing_CVPR_2021_paper.html](https://openaccess.thecvf.com/content/CVPR2021/html/Shi_Learning_by_Planning_Language-Guided_Global_Image_Editing_CVPR_2021_paper.html)
- 官方／作者 PDF：[https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_Learning_by_Planning_Language-Guided_Global_Image_Editing_CVPR_2021_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_Learning_by_Planning_Language-Guided_Global_Image_Editing_CVPR_2021_paper.pdf)
- 许可／访问：Copyright retained; open redistribution license not verified; local research copy only. Primary source catalogued; local PDF retrieval failed: curl: (56) schannel: server closed abruptly (missing close_notify)
- 本地副本：无；原因见下表。
- 版本／书目备注：CVF pages 13590-13599 preferred over Crossref 13585-13594; CVF author Trung Bui corrects Crossref Tmng Bui.

### [cie2020]

International Commission on Illumination (CIE). **CIE S 017:2020 ILV: International Lighting Vocabulary, 2nd edition (e-ILV)**. 2020. CIE S 017:2020。

- DOI: [10.25039/S017.2020](https://doi.org/10.25039/S017.2020)
- 官方／作者入口：[https://www.cie.co.at/e-ilv](https://www.cie.co.at/e-ilv)
- PDF：未确认免费完整标准 PDF；使用官方 HTML 术语页。
- 许可／访问：Official definitions readable online; full standard redistribution not verified. official HTML term pages available; no free official full PDF verified
- 本地副本：无；原因见下表。
- 版本／书目备注：Official online terminology; no freely downloadable full standard PDF verified.
- CIE colourfulness：[https://cie.co.at/eilvterm/17-22-072](https://cie.co.at/eilvterm/17-22-072)
- CIE saturation：[https://cie.co.at/eilvterm/17-22-073](https://cie.co.at/eilvterm/17-22-073)
- CIE chroma：[https://cie.co.at/eilvterm/17-22-074](https://cie.co.at/eilvterm/17-22-074)
- CIE brightness：[https://cie.co.at/eilvterm/17-22-059](https://cie.co.at/eilvterm/17-22-059)
- CIE lightness：[https://cie.co.at/eilvterm/17-22-063](https://cie.co.at/eilvterm/17-22-063)

## 11. 下载结果与不可获取清单

截至本次核查：**5 / 12 篇论文完整下载并解析；7 篇本地缺失；CIE 为额外的在线术语来源，无完整标准 PDF。** 所有本地副本仅用于本地研究，未上传、提交或再分发；安装包只使用书目元数据，不打包这些 PDF。文件完整性按 PDF 头、尾部 EOF、解析页数与 SHA-256 检查，不把部分下载计作成功。

| ID | 本地状态 | 本次原因／处理 |
|---|---|---|
| debevec1997 | 缺失 | curl: (28) Operation timed out after 150009 milliseconds with 674489 out of 1434221 bytes received；后续 .NET 请求也出现 TLS 连接失败。 |
| reinhard2002 | 已下载 | 352238 bytes；11 页；哈希见 manifest |
| peli1990 | 缺失 | curl: (28) Failed to connect to pelilab.partners.org:443 after 22967 ms: Could not connect to server；后续 .NET 请求也出现 TLS 连接失败。 |
| hasler2003 | 已下载 | 182038 bytes；9 页；哈希见 manifest |
| foi2008 | 缺失 | curl: (28) Operation timed out after 150006 milliseconds with 3468072 out of 6751576 bytes received；后续 .NET 请求也出现 TLS 连接失败。 |
| wang2004 | 已下载 | 1767707 bytes；13 页；哈希见 manifest |
| zhang2018 | 已下载 | 1712552 bytes；10 页；哈希见 manifest |
| fivek2011 | 缺失 | curl: (18) end of response with 1519988 bytes missing；后续 .NET 请求也出现 TLS 连接失败。 |
| acceptable2015 | 已下载 | 1637672 bytes；9 页；哈希见 manifest |
| exposure2018 | 缺失 | curl: (28) Operation timed out after 150012 milliseconds with 1504423 out of 6579154 bytes received；后续 .NET 请求也出现 TLS 连接失败。 |
| gier2021 | 缺失 | curl: (28) Operation timed out after 150014 milliseconds with 2925120 out of 34841158 bytes received；后续 .NET 请求也出现 TLS 连接失败。 |
| planning2021 | 缺失 | curl: (56) schannel: server closed abruptly (missing close_notify)；后续 .NET 请求也出现 TLS 连接失败。 |
| cie2020 | HTML 来源 | 未验证到免费完整标准 PDF；不以标准销售页冒充下载地址。 |

这份缺失清单说明本次网络可获取性，不推断全球不可获取、永久失效或出版社要求付费。没有绕过访问控制或关闭 TLS 验证；失败的临时下载已清除。Hasler 原旧式链接失败后，使用 EPFL 当前公开仓库 API 取得正式 bitstream；来源 URL 已更新。


## 集成修正记录（2026-10-26）

第 9 节核查问题已在 Agent 词表落实：暖冷引用改为 CIE 术语并标注 Lab b* 代理局限；反差补 Peli；鲜艳/淡雅补 CIE；锐化与降噪引文用途注明为评价或建模背景，未将其描述为现有算子的直接证据。
