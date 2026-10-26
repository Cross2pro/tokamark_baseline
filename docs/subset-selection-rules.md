# 子集选数规则（Subset Selection Rules）

适用于 Task 2 / 3 / 4 的固定子集实验（CNN vs TCN）。任何新子集必须遵守本文件。

## 术语

- **炮（shot）**：一次放电实验，对应一个 `<shot_id>.zarr` / `<shot_id>.zarr.zip`。
- **信号键（signal key）**：zarr 里的数组路径，形如 `gas_injection/total_injected`（注气总量）。
- **合格炮**：对该子集目标任务的**全部输入、执行器、输出**信号键，zarr 中都有对应数组（允许数组内有 NaN；**缺数组/缺变量不合格**）。

## Global Constraint（强制）

1. **信号齐全**：子集必须覆盖目标任务（默认 Task 2-1..2-3、3-1..3-3、4-1..4-5）配置里列出的全部输入 / 执行器 / 输出信号。缺失键的炮不得入选。
2. **选数前先扫描**：用 `scripts/scan_signal_coverage.py` 对 tar 内 `.zarr.zip` 做 namelist 覆盖扫描（不要为了扫描而解压 565GB）。
3. **固定划分**：选出后钉死炮号清单与 train/val/test；默认 `seed=23`，划分 **172 / 10 / 18**（若合格炮 <200，如实报告并选满，按比例划分）。
4. **来源可追溯**：必须记录每炮来自哪个 `batch_XXXXX.tar`（`fixed_shots_200_sources.csv`）。
5. **校验**：对清单与 split CSV 计算 sha256，写入 `fixed_shots_200_meta.json`。
6. **多样性**：合格池足够时，跨多个 tar round-robin 抽样，避免全集中在单一 batch。

## 标准产物路径

| 文件 | 含义 |
|------|------|
| `tokamark_baseline/src/config/fixed_shots_200.txt` | 钉死的 200 炮号 |
| `tokamark_baseline/src/config/fixed_split_200.csv` | 172/10/18 划分 |
| `tokamark_baseline/src/config/fixed_shots_200_sources.csv` | 炮 → tar/offset/size/split |
| `tokamark_baseline/src/config/fixed_shots_200_meta.json` | 元数据 + sha256 |
| `tokamark_baseline/src/config/fixed_shots_200_batch00000_legacy.txt` | 旧 batch_00000 清单备份（信号不齐，仅 legacy） |
| `logs/signal_scan/` | 扫描结果（`required_signals_*`, `scan_coverage.csv`, `qualified_shots*`） |

工作区根目录会同步一份同名文件，供当前 subset YAML 的 `data_splits_file` 引用。

## 数据落地

优先只抽出这 200 个 `.zarr.zip` 到：

`/home/remote/lee/datasets/MAST-zarrzip-fixed200/`

训练配置 `store_manager_settings.base_local_zarr_path` 指向该目录（代码已支持 `.zarr.zip`）。旧 `MAST-zarr`（batch_00000 解压目录）保留不删，标为 legacy。

## 命令

```bash
cd /home/remote/lee/gh-pr/tokamark_workspace
# 1) 扫描
nice -n 19 ionice -c3 tokamark-venv313/bin/python -u scripts/scan_signal_coverage.py \
  --out-dir logs/signal_scan
# 2) 选 200
tokamark-venv313/bin/python -u scripts/select_fixed200.py --seed 23 --n 200
# 3) 抽出 zip
scripts/extract_fixed200_zips.sh
```

## 反例（不要再犯）

- 仅用 `batch_00000.tar` 前 200 炮：普遍缺少 `gas_injection/total_injected`、部分缺少 Thomson / soft X-ray / Mirnov 相关键，**不能**作为 Task 2/3/4 规范子集。
