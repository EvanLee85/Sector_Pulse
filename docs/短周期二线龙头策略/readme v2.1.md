# 短周期二线龙头量化交易系统 · README【v2.1.2 · 严格对齐版】

> 🎯 面向 A 股散户的 T+1 ~ T+5 短周期自动化交易系统  
> 以“闸门—阈值—剧本”为骨架，融合 **QVIX** 波动治理、**SOP** 风控、**交通灯**全局可观测体系

[![Python](https://img.shields.io/badge/python-3.11-blue)](#)
[![AkShare](https://img.shields.io/badge/akshare-latest-green)](#)
[![Streamlit](https://img.shields.io/badge/streamlit-latest-red)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-orange)](#)

---

## 🧭 对齐基准
- **00 策略**：短周期二线龙头策略 · V2.3.1（QVIX 版，RR/Pwin/EV 闸门）
- **01 流程图**：mermaidchart.com 流程图 v3.3（**蓝图**）
- **02 项目结构**：项目文件结构 v1.3.1（**标准：命名/落位/脚本以此为准**）
- **开发计划**：v2.1.2（本 README 与开发计划完全一致）

> 口径约定：**任何新增功能先对齐 02 的命名/落位**；流程图为蓝图，通过“**映射表**”对应到稳定接口。

---

## 🏗️ 系统架构（含数据接口抽象层）

```
┌───────────────────────────────┐
│        Web 前端（Streamlit）   │  dashboard / pages / components
└──────────────┬────────────────┘
               │
        ┌──────▼──────┐
        │  策略引擎层  │  regime / sector & concept / selector / signals /
        │              │  ev / risk / portfolio / adaptive weights
        └──────┬──────┘
               │
        ┌──────▼────────────────────────────────────────────┐
        │  数据服务层（**抽象接口**）                        │
        │  data_service/collector.py::DataCollector         │
        │  ⚠️ 所有数据访问统一走此入口，**禁止直连 akshare**     │
        └──────┬────────────────────────────────────────────┘
               │    API 映射 / 缓存 / 限流 / 熔断 / 备用源 / Mock
               ▼
        config/data_source_mapping.json（配置驱动）
```

### 🔴 开发红线（**务必遵守**）
- **统一入口**：所有采集调用均经 `DataCollector.*`；除 `data_service/collector.py` 外**禁止** `import akshare|eastmoney`
- **配置驱动**：新增/切换数据源，通过 `config/data_source_mapping.json` 映射；对外函数名 **不变**
- **命名对齐**：文件/类/函数与 **02** 完全一致（`snake_case` / `PascalCase`）

---

## 🚀 快速开始

### 环境
- Python 3.11（推荐） / 2GB+ RAM / 稳定网络 / `Asia/Shanghai` 时区

### 安装
```bash
git clone https://github.com/your-username/trading-system.git
cd trading-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 初始化数据库（按 02 标准脚本）
python scripts/init_database.py

# （可选）导入历史数据（Demo/冷启动友好）
python scripts/import_historical_data.py --days 180
```

### 启动方式
```bash
# 完整系统（前端 + 调度）
python main.py --all

# 仅前端
python main.py --web

# 仅调度（建议用此跑盘前/盘后）
python main.py --scheduler
```

**✅ 安装自检清单**
- [ ] `python -c "import pytz;import datetime as dt;print(dt.datetime.now(pytz.timezone('Asia/Shanghai')))"` 显示北京时间  
- [ ] `python scripts/system_check.py` 输出 **绿色**（见“系统健康阈值”）

---

## 🔧 配置说明

- `config/settings.py`：数据库/日志/调度/Web/通知/时区（`TIMEZONE="Asia/Shanghai"`）/特性开关
- `config/data_source_mapping.json`：**数据源映射与策略**（主/备源、缓存 TTL、限流、重试、熔断、Mock）
- `config/default_params.json` / `user_params.json`：策略阈值与个人化参数（含 `"source":"00/章节/表格名"` 注释）
- `config/param_schema.json` / `config/param_history.json`：参数校验与版本回滚

**最小映射示例（节选）**
```json
{
  "get_basic_market_data": {
    "primary": "akshare.stock_zh_a_spot_em",
    "fallbacks": ["mock.get_mock_market_data"],
    "retry": {"max": 3, "backoff": 2.0},
    "rate_limit": {"rpm": 60},
    "cache_ttl": 60
  },
  "get_risk_indicator_data": {
    "compose": {
      "steps": [
        {"name": "qvix_proxy_from_options_minute"},
        {"name": "a50_sgx_settlement"},
        {"name": "margin_sh_sse"}
      ]
    },
    "fallbacks": ["mock.get_risk_indicator_data_daily"],
    "cache_ttl": 180
  }
}
```

---

## 🧩 数据接口抽象层（DataCollector）· **7/7 稳定接口 + 2 个采集口**

**稳定接口（以 02 为准，唯一标准）**
- `get_basic_market_data()`  
- `get_sector_data()`  
- `get_correlation_data()`（计算型，读库聚合）  
- `get_fund_flow_data()`  
- `get_risk_indicator_data()`（内部可组合 QVIX/A50/两融）  
- `get_dragon_tiger_data()`  
- `get_etf_flow_data()`  
- `get_extended_data(data_type: str)`（扩展口）

**点名采集口（仅这两个需要存在）**
- `collect_a50_futures()`  
- `collect_margin_data()`

**容错/缓存/热更新**
- 三级缓存：内存（短 TTL）→ 磁盘（中 TTL）→ API  
- 熔断与降级：连续失败→短时熔断→备源回退→Mock  
- 数据年龄闸门：`data_age` 超阈值触发降级/告警  
- 热更新：监听 `data_source_mapping.json` 的 mtime 自动重载；`FEATURE_FLAGS` 动态开关

**✅ 抽象层自检**
- [ ] 7/7 `get_*` + `collect_a50_futures/margin` **同名存在**  
- [ ] `pytest -q tests/test_data_interface.py::test_all_interfaces_exist` 通过  
- [ ] 仅 `collector.py` 出现 `import akshare|eastmoney`（pre-commit/CI 拦截其余路径）

---

## 🟢🟡🔴 交通灯状态体系（可观测 · 可联动 · 可验收）

> 目的：一眼看懂“能不能跑”、“要不要降级”、“何时休眠”。  
> 规则：**任何一级出现🔴，全局置🔴并触发休眠/降档**；出现🟡则进入“降级模式”，继续运行但功能/门槛降级。  
> 关键时间门（盘前 09:10 前完成）：`Regime` 计算、**三路预测信号**（龙虎榜/ETF/PCR）入库、指标更新。  
> 前端：`dashboard` 顶部显示三灯条；各页面卡片显示分灯。

### 1) 数据源与抽象层（A0）
| 模块 | 🟢 绿色 | 🟡 黄色（降级） | 🔴 红色（阻断） | 查看/操作 |
|------|---------|----------------|-----------------|----------|
| DataCollector | `error_rate<1%` 且 `p50_latency<500ms` 且 `data_age≤2min` | `1%≤error_rate<5%` 或 `2min<data_age≤10min`（启用缓存/备源） | `error_rate≥5%` 或 `data_age>10min`（熔断+MOCK） | `python scripts/system_check.py`；Overview「数据灯」 |

**联动**：🟡→自动 fallback/提高 `cache_ttl`；🔴→`Notification.send_alert()` + Risk Monitor 顶部阻断横幅。

### 2) 盘前流程（采集→清洗→指标）
| 模块 | 🟢 | 🟡（降级） | 🔴 | 查看/操作 |
|------|----|------------|-----|----------|
| 采集/清洗/入库 | 全部成功；概念四口径均有新增 | 任一源重试成功、或概念表缺少其一（history_min 可延后） | 关键源失败且无备源：`sector/qvix/dragon_tiger/etf` 任两项以上缺失 | `--job premarket` 日志；Overview 数据块 |
| 指标 | QVIX 分位/日变入库成功 | 使用昨日缓存（标注“昨日缓存”） | 今日指标缺失 | Performance「指标灯」 |

**联动**：09:10 前若 **PCR/龙虎榜/ETF** 不齐 → 🟡；若两路以上缺失 → 🔴，禁用买点生成。

### 3) Regime 判定（六状态→四段位）
|  模块  |           🟢          |            🟡（降级）       |            🔴         |        查看/操作          |
|--------|-----------------------|-----------------------------|------------------------|--------------------------|
| Regime | 今日重算且软/硬条件有效 | 使用昨日 Regime（标注“昨日”） | QVIX/A50/两融不足以判定 | Risk Monitor「Regime 灯」 |

**联动**：🔴→**系统休眠**；🟡→段位强制下调 1 档，Kelly 上限下调 30%。

### 4) 预测信号（F2，三路为硬依赖）
| 模块 | 🟢 | 🟡（降级） | 🔴 | 查看/操作 |
|---|---|---|---|---|
| 龙虎榜 | 当日齐全 | 使用 T-1 | 今日缺失且无备源 | Signals「数据灯」 |
| ETF 异动 | 分钟级 | 日级代理 | 数据缺失 | 同上 |
| 期权 PCR | 分钟级 | 回退到日级 | 无法计算 | 同上 |

**联动**：任一🟡→降级黄灯（权重降低/置信区间放宽）；≥1 路🔴→禁用买点生成（仅出候选）。

### 5) 选股与买点（F1/F3）
| 模块 | 🟢 | 🟡（降级） | 🔴 | 查看/操作 |
|---|---|---|---|---|
| 候选池 | 1–5 只 | 0 或 >5（阈值自动调） | 无法生成 | Stock Selection |
| 买点生成 | **RR≥2.0** 且 Pwin≥0.60 且 EV_net>0 | 边界（RR∈[1.8,2.0) 或 Pwin∈[0.55,0.60)） | 任一闸门失败 | Signals「买点灯」 |

**联动**：🟡→下调仓位上限 30%；🔴→仅影子单。

### 6) EV 与风控（G）
| 模块 | 🟢 | 🟡（降级） | 🔴 | 查看/操作 |
|---|---|---|---|---|
| EV 评估 | EV_net>0 且路径依赖指标正常 | 接近 0（±5%） | EV_net≤0 | Performance / Risk Monitor |
| SOP 红线 | 未触发 | 告警（接近红线） | 触发（回撤/波动/流动性） | Risk Monitor「红线灯」 |

**联动**：🔴→`trigger_sleep_mode()`；🟡→`execute_shadow_trading()` 并冻结加因子。

### 7) 执行与日志
| 模块 | 🟢 | 🟡（降级） | 🔴 | 查看/操作 |
|---|---|---|---|---|
| 执行链 | 成功率≥95% | 80–95% | <80% 或连续失败≥3 | Trade Log；执行监控卡 |

**🧪 交通灯验收步骤**
1. 运行：`python main.py --scheduler --job premarket`；`python scripts/system_check.py`  
2. 前端观察：Dashboard 三灯条；Risk Monitor（Regime/红线/横幅）；Overview（数据灯）；Signals（三路+买点）  
3. 降级回归：临时关闭一个主源或强制报错≥3，观察状态从 🟢→🟡/🔴 的日志与 UI  
4. 自动化断言：`tests/test_lights.py` 覆盖 **全绿**/**降级**/**阻断**

> RR 基线：**默认 RR≥2.0**；*注：进攻-加速子状态可按样本放宽到 1.8，但默认仍以 2.0 为基线。*

---

## 📊 核心功能摘要（与 00/01/02 对齐）

- **Regime 状态识别**：QVIX 接管闸门（分位 & 日变）；四段位（进攻/标准/防守/休眠）  
- **板块轮动「六步+预测」**：强弱/广度/延续/资金/背书/ETF；预测= **龙虎榜+ETF 异动+PCR（必做）**  
- **动态选股（四层漏斗）**：流动性→RS→基本面避雷→技术确认；日候选 1–5  
- **信号与统一闸门**：突破/回踩/趋势/反包；RR≥2.0、Pwin≥0.60、EV_net>0  
- **路径依赖 EV**：隔夜跳空/盘中回撤/时间衰减/冲击成本 + Monte Carlo  
- **仓位与剧本**：Kelly×保守系数；T+5 强制平仓；利润回撤触发剧本  
- **SOP 风控**：三红线（回撤/QVIX/流动性）+ 两退化（EV/Pwin 退化）→ 休眠/降档

---

## 🛠️ 运维与排障（脚本与 02 对齐）

```bash
# 系统管理
python scripts/start_system.py
python scripts/stop_system.py

# 初始化 / 备份 / 恢复 / 清理
python scripts/init_database.py
python scripts/backup_data.py
python scripts/restore_data.py
python scripts/cleanup.py

# 检查与报表
python scripts/system_check.py
python scripts/generate_report.py

# （可选）导入历史数据
python scripts/import_historical_data.py --days 180
```

**✅ 运维自检**
- [ ] 非交易日执行 `--scheduler` 仅跑维护/报告  
- [ ] `system_check.py` 输出 **数据/缓存/延迟/Schema/备份** 五项指标

---

## 🕒 时区与交易日历（A 股专用）

- 默认时区：**Asia/Shanghai**（`config/settings.py` → `TIMEZONE="Asia/Shanghai"`）  
- 交易日历：`utils.helpers.get_trade_calendar()`（休市/节假日自动降级为维护任务）  
- 关键时间门：盘前 **09:10** 完成 Regime 与三路预测信号（龙虎榜/ETF/PCR）

**✅ 验收**
- [ ] 休市日跑 `--job premarket` 不下单，仅维护/报告  
- [ ] `system_check.py` 中“Calendar/Timezone”项为绿色

---

## 📦 冷启动与离线跑通（无网/源挂也能演示）

- 启用 Mock：`FEATURE_FLAGS={"mock": true}` 或在映射中将 `fallbacks` 指向 `mock.*`  
- Demo 数据：`python scripts/import_historical_data.py --demo`  
- 最小链路：`python demo_min_flow.py` → Overview 显示真实表格

**✅ 验收**
- [ ] 断网下 `system_check.py` 显示「备源/Mock 生效」，前端可展示  
- [ ] `logs/` 出现「fallback/mock」关键词

---

## 🛌 休眠模式（Sleep Mode）行为定义

- 触发：SOP 红线，或 Regime=休眠，或**交通灯全局🔴**  
- 行为：**停止买点与仓位更新**；保留 **采集/指标/报告**；`trade_executor` 仅写影子单  
- 解除：观察期满足条件（如 3 个交易日未触红线且 QVIX < 75 分位）

**✅ 验收**
- [ ] 休眠后 `pages/02_stock_selection.py` 不显示买点，`pages/04_risk_monitor.py` 显示横幅  
- [ ] `trigger_sleep_mode()` 执行后，`scheduler` 不再触发下单

---

## 🗃️ 数据库版本与迁移

- 版本表：`schema_version(version, applied_at, checksum)`  
- 基线：`init_database.py` 写入 `version=1`  
- 迁移脚本：`scripts/migrate_###.py --upgrade/--downgrade`

**✅ 验收**
- [ ] `SELECT * FROM schema_version ORDER BY applied_at DESC LIMIT 1;` 能看到当前版本  
- [ ] 升/降级后 `system_check.py` 的「Schema」项为绿色

---

## 🧹 数据留存与归档策略

- 分钟级行情：保留 **90 天**；日线永久  
- 概念四口径历史：永久（归档至 `data/archive/`）  
- 索引：为 `timestamp,symbol`、`board_type,board_id,date` 建复合索引  
- 周备份：`backup_data.py`；月清理：`cleanup.py`

**✅ 验收**
- [ ] 周备份后 `backups/` 有新包  
- [ ] 清理后 DB 体量下降、查询性能稳定

---

## 🧭 蓝图→标准 映射表入口（不扩口）

- 文档：`docs/flow_to_api_mapping.md`  
  - 例：`collect_market_data → get_basic_market_data`  
        `collect_sector_data → get_sector_data`  
        `collect_qvix_data → get_risk_indicator_data(QVIX 分支)`  
        `collect_a50_futures → collect_a50_futures`（同名口保留）  
        `collect_margin_data → collect_margin_data`（同名口保留）

**✅ 验收**
- [ ] `scheduler/premarket_tasks.py` 注释引用该表  
- [ ] 新流程节点仅改文档/映射，不新增对外函数

---

## 🛡️ 红线（禁直连）安装与本地校验

```bash
pip install pre-commit
pre-commit install
# 本地显式触发一次
pre-commit run --all-files
```

**✅ 验收**
- [ ] 除 `collector.py` 外，任意文件 `import akshare|eastmoney` 会 **提交失败**  
- [ ] `pytest -q tests/test_data_interface.py::test_no_direct_api_calls` 通过

---

## 🧪 参数追溯（源自 00 的阈值）

- 在 `config/default_params.json` 的关键阈值上标注注释：`"source": "00/章节/表格名"`  
- 差异报告：`python tools/param_diff.py --from 00 --to config/default_params.json`

**✅ 验收**
- [ ] `reports/param_diff_*.md` 生成且差异项完整  
- [ ] README 中 RR/Pwin/EV 阈值与参数文件一致（默认 **RR≥2.0**）

---

## 📈 系统健康阈值（对应交通灯🟢的数字口径）

- 抽象层：`error_rate < 1%`、`p50_latency < 500ms`、`data_age ≤ 2min`  
- 盘前：**09:10 前**三路预测信号（龙虎榜/ETF/PCR）+ 指标入库齐全  
- 执行：任务成功率 ≥ 95%

**✅ 验收**
- [ ] `python scripts/system_check.py` 输出三项指标均为绿色  
- [ ] Dashboard 三灯条与数值一致

---

## ❓FAQ / 错误码（示例）

- **E001 数据源失败**：检查 `data_source_mapping.json` → 将 `primary.enabled=false` 验证 fallback 是否接管  
- **E002 QVIX 无法计算**：回退到**日级 PCR** 与 **ETF 异动**；`aggregate_signals()` 降级  
- **E003 前端空页面**：先跑 `python main.py --scheduler --job premarket`，再开 `python main.py --web`  
- **E004 直连被拦截**：安装 `pre-commit` 并修复违规 `import`；仅允许在 `collector.py`  

---

## 📄 许可证与免责声明
- MIT License  
- 本项目仅用于学习研究，不构成投资建议；实盘前务必在模拟环境充分验证

## 🙏 致谢
AkShare / Streamlit / Pandas / APScheduler 等开源项目
