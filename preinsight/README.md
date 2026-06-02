# preinsight

Day-0.5 预实验脚本:在烧 GPU 之前,先确认 DriveLM 里"看不见的事实(遮挡/未来/地图)驱动了减速/避让动作"的场景到底有多少 —— 也就是能用来拼 naive / 干净 两套解释监督的场景够不够。

## drivelm_coverage.py

逐帧扫 DriveLM 标注,统计:
- `has_occlusion / has_future / has_map`:解释里是否提到部署时看不见的事实
- `has_brake_action`:是否在减速/避让
- `usable`:三类看不见事实之一 **且** 有减速动作

默认**纯规则(关键词)统计**,无需 GPU、无需联网、仅标准库。可选用本地 GLM 抽样校验规则数得准不准。

### 用法
```bash
# 纯统计(几十秒):
python3 drivelm_coverage.py --drivelm /path/to/v1_0_train_nus.json --out ./coverage_out

# 可选:本地 GLM 抽样校验(先改脚本顶部 GLM_BASE_URL 里的 __GLM_IP__)
python3 drivelm_coverage.py --drivelm .../v1_0_train_nus.json --out ./coverage_out --use-glm
```

### 输出
- `coverage_out/coverage_summary.md` —— 总数 + 各类占比 + go/no-go 判断 + 样例
- `coverage_out/coverage_per_frame.csv` —— 逐帧明细

### 判断档(脚本自动给)
- usable ≥ 300 → 信号充足,继续写 naive/干净 生成脚本
- 50–300 → 够做预检,正式实验可能要补 Bench2Drive/SimLingo
- < 50 → 太稀,DriveLM 单独撑不起,换/补数据集

### 依赖
仅 Python3 标准库(json / urllib / argparse / collections / random / re),无需 pip install。
GLM 走 OpenAI 兼容接口,改脚本顶部 `GLM_BASE_URL / GLM_MODEL` 即可。
