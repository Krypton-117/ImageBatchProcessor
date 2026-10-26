"""Transparent lexical checks for explicit requests; model output remains in the audit."""

import re

PATTERNS = {
    "brighten": r"亮一点|亮一些|亮些|提亮|提高亮度|增加曝光|明亮点|太暗|偏暗|曝光.{0,4}(加|提高|增加)",
    "darken": r"暗一点|暗一些|暗些|压暗|降低亮度|减少曝光",
    "highlights": r"(压|降|收|救).{0,6}(高光|亮部)|(高光|亮部).{0,6}(压|降|收)|刺眼",
    "shadows": r"(提|亮).{0,6}暗部|暗部.{0,6}(提|亮)|阴影.{0,6}(提|亮)|暗处.{0,6}(提|亮)|增加.{0,4}暗部层次",
    "contrast": r"通透|反差大|增加反差|增强对比|灰蒙蒙|灰扑扑",
    "soft": r"柔和|反差小|降低反差|降低对比|反差别那么强",
    "warm": r"暖一点|暖一些|暖些|暖和|偏暖|太冷|发蓝",
    "cool": r"冷一点|冷一些|冷些|偏冷|发黄|太黄",
    "vivid": r"鲜艳|更艳|艳一点|增加饱和|色彩浓(一点|一些|些)",
    "muted": r"淡雅|降低饱和|饱和度低|淡一点|淡一些|褪色|太艳|色彩浓度.{0,3}降",
    "denoise": r"降噪|去噪|噪点|噪声|干净|颗粒",
    "sharpen": r"锐化|清晰|清楚|锐利|有点糊|模糊",
    "natural": r"自然|原来的感觉",
}
PRESERVE = {
    "color": r"(不要|别|不想|不能).{0,5}(太艳|更艳|增加饱和|变艳)|保持.{0,5}(色彩|颜色|饱和)|保留.{0,5}(色彩|颜色)",
    "brightness": r"(保持|保留).{0,5}亮度|亮度.{0,4}(不变|不要动)",
    "warmth": r"(保持|保留).{0,8}(暖色|冷色|色温)|色温.{0,4}(不变|不要动)",
    "highlights": r"(别|不要).{0,6}(灯|高光).{0,4}(亮|爆)|保留高光",
    "details": r"(保持|保留|别抹|不要抹).{0,5}细节|细节.{0,4}(保留|不变)",
}
UNSUPPORTED = r"(只|仅).{0,8}(脸|人脸|天空|背景|人物|主体)|删除|去掉.{0,4}(人|物)|换.{0,3}(背景|天空)|生成|加.{0,3}(物体|人物)|磨皮|瘦脸|抠图|局部|执行命令|(衣服|衣物|头发).{0,8}(变成|换成|改成|染成)|换色"


def guard_intent(prompt, raw):
    result = {k: (list(v) if isinstance(v, list) else v) for k, v in raw.items()}
    clauses = [
        x.strip()
        for x in re.split(r"[，,。；;！!\n]|但是|但|而且", prompt)
        if x.strip()
    ]
    explicit = []
    preserve = []
    negated = []
    unknown = []
    for clause in clauses:
        kept = [k for k, p in PRESERVE.items() if re.search(p, clause)]
        preserve.extend(kept)
        if kept:
            continue
        hits = [k for k, p in PATTERNS.items() if re.search(p, clause)]
        if (
            re.search(r"不要|不想|别|不能", clause)
            and not re.search(r"反差别那么强", clause)
            and not re.search(
                r"(噪点|噪声|颗粒).*(不要|别)|不要.*(噪点|噪声|颗粒)", clause
            )
        ):
            negated.extend(hits)
        else:
            explicit.extend(hits)
        if not hits:
            unknown.append(clause)
    blocked = bool(re.search(UNSUPPORTED, prompt))
    # An undersized model must not silently guess unfamiliar or aesthetic-only requests.
    if not unknown and explicit and not blocked:
        result.update(
            intents=list(dict.fromkeys(explicit)),
            preserve=list(dict.fromkeys(preserve)),
            unsupported=False,
            clarification=False,
        )
    else:
        result["intents"] = list(dict.fromkeys(result["intents"] + explicit))
        result["preserve"] = list(dict.fromkeys(result["preserve"] + preserve))
    result["intents"] = [
        x
        for x in result["intents"]
        if x not in negated and not (x == "vivid" and "color" in preserve)
    ]
    if blocked:
        result["unsupported"] = True
    meaningful_unknown = [
        x
        for x in unknown
        if not re.fullmatch(r"(请|谢谢|别太夸张|适当就好|轻微一点)", x)
    ]
    if meaningful_unknown and not blocked:
        result["clarification"] = True
    if re.search(r"([0-9半一二两三四五六七八九十]+\s*(档|EV|ev|%|K))", prompt):
        result["clarification"] = True
    changes = {
        k: {"model": raw[k], "guarded": result[k]} for k in raw if raw[k] != result[k]
    }
    result["raw_model_result"] = raw
    result["language_guards"] = {
        "changes": changes,
        "unmatched_clauses": unknown,
        "basis": "explicit lexical/negation rules, not additional model inference",
    }
    return result
