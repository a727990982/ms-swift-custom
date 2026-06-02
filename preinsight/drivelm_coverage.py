#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DriveLM 覆盖度统计 (Day-0.5)
回答一个问题:DriveLM 里到底有多少"遮挡 / 未来 / 地图 事实驱动了一个减速/避让动作"的场景?
—— 也就是我们能用来拼 naive/干净 两套解释的场景到底够不够。

设计:
  1) 规则解析(默认,纯标准库,离线即可跑):按类别 + 关键词,逐帧标记
     has_occlusion / has_future / has_map / has_brake_action,
     并算出 "usable"(三类事实之一 AND 一个减速/避让动作)。
  2) GLM 抽样校验(可选):随机抽 N 帧,让本地 GLM 判一遍,和规则比对,
     给出一致率 —— 用来判断规则数得准不准。

依赖:仅 Python3 标准库(json/urllib/argparse/collections/random/re)。无需 pip install。

用法:
  python3 drivelm_coverage.py --drivelm /path/to/v1_0_train_nus.json --out ./coverage_out
  # 想开 GLM 校验:把下面 USE_GLM 改 True、填好 GLM_BASE_URL,再加 --use-glm
"""

import argparse
import json
import os
import random
import re
import urllib.request
from collections import Counter

# ============================ 改这里(GLM 接入) ============================
# 你的本地 GLM 服务地址(OpenAI 兼容接口)。把 __GLM_IP__ 换成实际 IP / 端口。
GLM_BASE_URL = "http://__GLM_IP__:8000/v1"   # 例: http://192.168.1.50:8000/v1
GLM_MODEL    = "glm-4"                         # 你部署的模型名
GLM_API_KEY  = "EMPTY"                         # 本地 vLLM/服务通常随便填
USE_GLM      = False                           # True 才调用 GLM 做抽样校验(也可用 --use-glm 命令行开)
GLM_SAMPLE_N = 50                              # 抽多少帧让 GLM 校验
GLM_TIMEOUT  = 60
# =========================================================================

# ---- 关键词表(可按需增删) ----
KW_OCCLUSION = ["occlud", "blocked by", "behind the", "behind it", "hidden",
                "not visible", "out of sight", "obstruct", "obscur"]
KW_FUTURE    = ["future", "will ", "going to", "about to", "next few", "next second",
                "moving to", "move to", "cross the", "is going", "predict", "intend"]
KW_MAP       = ["speed limit", "traffic light", "change lane", "lane change",
                "right of way", "intersection", "stop sign", "traffic sign",
                "lane marking", "which direction can", "allowed to"]
KW_BRAKE     = ["brake", "braking", "slow down", "slowing", "stop", "stopping",
                "decelerat", "yield", "wait", "keep distance", "cautious",
                "give way", "reduce speed"]


def text_of(qa_list):
    """把一个类别里所有 Q+A 拼成一段小写文本,便于关键词匹配。"""
    out = []
    for item in qa_list or []:
        q = str(item.get("Q", ""))
        a = str(item.get("A", ""))
        out.append(q + " " + a)
    return " \n ".join(out).lower()


def hit(text, kws):
    return any(k in text for k in kws)


def iter_frames(data):
    """兼容 DriveLM-nuScenes 结构: {scene_token: {"key_frames": {frame_token: {...}}}}。
    退化兼容: 顶层直接是 {frame_token: {...}} 的情况。"""
    if not isinstance(data, dict):
        return
    for scene_token, scene in data.items():
        if isinstance(scene, dict) and "key_frames" in scene:
            for frame_token, frame in scene["key_frames"].items():
                yield scene_token, frame_token, frame
        elif isinstance(scene, dict) and "QA" in scene:
            yield "?", scene_token, scene


def classify_frame(frame):
    """规则解析一帧,返回各标记。"""
    qa = frame.get("QA", {}) or {}
    perc = text_of(qa.get("perception"))
    pred = text_of(qa.get("prediction"))
    plan = text_of(qa.get("planning"))
    beh  = text_of(qa.get("behavior"))
    all_fact = perc + " \n " + pred                 # 事实层(感知+预测)
    all_act  = plan + " \n " + beh                  # 动作层(规划+行为)

    has_occ = hit(all_fact, KW_OCCLUSION)
    # 未来:prediction 类别天然偏未来,再用关键词收一下(降低误报)
    has_fut = hit(pred, KW_FUTURE) or hit(all_fact, KW_FUTURE)
    has_map = hit(all_fact + " " + all_act, KW_MAP)
    has_brk = hit(all_act, KW_BRAKE)

    usable = (has_occ or has_fut or has_map) and has_brk
    return {
        "has_occlusion": has_occ,
        "has_future": has_fut,
        "has_map": has_map,
        "has_brake_action": has_brk,
        "usable": usable,
        "n_perception": len(qa.get("perception") or []),
        "n_prediction": len(qa.get("prediction") or []),
        "n_planning":  len(qa.get("planning") or []),
        "n_behavior":  len(qa.get("behavior") or []),
    }


# ----------------------- 可选: GLM 抽样校验 -----------------------
def call_glm(prompt):
    body = {"model": GLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0}
    req = urllib.request.Request(
        GLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + GLM_API_KEY},
    )
    with urllib.request.urlopen(req, timeout=GLM_TIMEOUT) as r:
        resp = json.loads(r.read().decode("utf-8"))
    return resp["choices"][0]["message"]["content"]


GLM_PROMPT = """You audit a driving scene's QA annotations for "deploy-unobservable" facts.
A deploy-clean model has ONLY the current camera. Decide if the QA below references any fact that
such a model could NOT observe from the camera, in three buckets:
- occlusion: a specific object hidden/occluded/behind another object.
- future: an event that has not happened yet (will move / will cross / next seconds).
- map: a fact only from HD map / regulation (speed limit, lane topology, allowed lane change, traffic-rule).
Return STRICT JSON only: {{"occlusion": true/false, "future": true/false, "map": true/false}}

QA (perception + prediction):
{fact_text}
"""


def glm_check(frame):
    qa = frame.get("QA", {}) or {}
    fact_text = (text_of(qa.get("perception")) + "\n" + text_of(qa.get("prediction")))[:4000]
    try:
        raw = call_glm(GLM_PROMPT.format(fact_text=fact_text))
        m = re.search(r"\{.*\}", raw, re.S)
        j = json.loads(m.group(0)) if m else {}
        return {"occlusion": bool(j.get("occlusion")),
                "future": bool(j.get("future")),
                "map": bool(j.get("map"))}
    except Exception as e:                       # 离线/服务异常时不崩,标记失败
        return {"_error": str(e)}


# ----------------------------- 主流程 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivelm", required=True, help="DriveLM v1_0_train_nus.json 路径")
    ap.add_argument("--out", default="./coverage_out", help="输出目录")
    ap.add_argument("--use-glm", action="store_true", help="开启 GLM 抽样校验")
    ap.add_argument("--glm-n", type=int, default=GLM_SAMPLE_N)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    use_glm = args.use_glm or USE_GLM
    os.makedirs(args.out, exist_ok=True)

    with open(args.drivelm, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    agg = Counter()
    n_frames = 0
    for scene_token, frame_token, frame in iter_frames(data):
        n_frames += 1
        c = classify_frame(frame)
        c["scene_token"] = scene_token
        c["frame_token"] = frame_token
        rows.append(c)
        for k in ["has_occlusion", "has_future", "has_map", "has_brake_action", "usable"]:
            if c[k]:
                agg[k] += 1

    if n_frames == 0:
        print("!! 没解析到任何 frame —— 检查 json 结构是否为 DriveLM-nuScenes 格式。")
        return

    def pct(k):
        return 100.0 * agg[k] / n_frames

    # ---- 写 per-frame csv ----
    csv_path = os.path.join(args.out, "coverage_per_frame.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["scene_token", "frame_token", "has_occlusion", "has_future",
                "has_map", "has_brake_action", "usable",
                "n_perception", "n_prediction", "n_planning", "n_behavior"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # ---- GLM 抽样校验 ----
    glm_report = ""
    if use_glm:
        rnd = random.Random(args.seed)
        sample = rnd.sample(rows, min(args.glm_n, len(rows)))
        # 用 frame_token 找回 frame 对象
        frame_lookup = {ft: fr for _, ft, fr in iter_frames(data)}
        agree = Counter(); total = Counter(); errs = 0
        for r in sample:
            fr = frame_lookup.get(r["frame_token"])
            if fr is None:
                continue
            g = glm_check(fr)
            if "_error" in g:
                errs += 1
                continue
            for bucket, rule_key in [("occlusion", "has_occlusion"),
                                     ("future", "has_future"),
                                     ("map", "has_map")]:
                total[bucket] += 1
                if g[bucket] == r[rule_key]:
                    agree[bucket] += 1
        lines = ["## GLM 抽样校验 (n=%d, 失败=%d)" % (len(sample), errs)]
        for b in ["occlusion", "future", "map"]:
            if total[b]:
                lines.append("- %s: 规则 vs GLM 一致率 = %.1f%% (%d/%d)" %
                             (b, 100.0 * agree[b] / total[b], agree[b], total[b]))
        lines.append("> 一致率低 = 规则关键词需要调,或该类太主观,数出来的覆盖度别全信。")
        glm_report = "\n".join(lines)

    # ---- 写 summary.md ----
    examples = [r for r in rows if r["usable"]][:8]
    md = []
    md.append("# DriveLM 覆盖度统计")
    md.append("")
    md.append("- 总 key-frame 数: **%d**" % n_frames)
    md.append("")
    md.append("| 标记 | 帧数 | 占比 |")
    md.append("|---|---:|---:|")
    for k, name in [("has_occlusion", "含遮挡事实"),
                    ("has_future", "含未来事实"),
                    ("has_map", "含地图/规则事实"),
                    ("has_brake_action", "含减速/避让动作"),
                    ("usable", "**可用**(三类事实之一 AND 减速动作)")]:
        md.append("| %s | %d | %.1f%% |" % (name, agg[k], pct(k)))
    md.append("")
    md.append("## 判断")
    if agg["usable"] >= 300:
        md.append("- 可用场景 **%d** 条 → 训练信号**充足**,可继续写 naive/干净 生成脚本。" % agg["usable"])
    elif agg["usable"] >= 50:
        md.append("- 可用场景 **%d** 条 → **偏少但够做预检**;正式实验可能要补 Bench2Drive/SimLingo。" % agg["usable"])
    else:
        md.append("- 可用场景仅 **%d** 条 → **太稀**,DriveLM 单独撑不起,考虑换/补数据集。" % agg["usable"])
    md.append("")
    md.append("## 可用场景样例(前 8 条 frame_token)")
    for r in examples:
        md.append("- %s | occ=%s fut=%s map=%s" %
                  (r["frame_token"], r["has_occlusion"], r["has_future"], r["has_map"]))
    if glm_report:
        md.append("")
        md.append(glm_report)

    md_path = os.path.join(args.out, "coverage_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))
    print("\n[写出] %s\n[写出] %s" % (md_path, csv_path))


if __name__ == "__main__":
    main()
