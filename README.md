# SectorPulse – 状态驱动的短周期量化交易系统

SectorPulse 是针对中国 A 股散户开发的半自动化交易框架，旨在提供一个稳定、可恢复且风险可控的 T+1～T+5 短周期交易系统。该项目基于 V3.1 策略蓝图，采用状态机驱动的决策模式，配合 “Alt‑Flow” 替代口径实现情绪闸门和 Kill‑Switch 风控，既支持最强板块候选股筛选，又提供完整的持仓管理和数据抽象层。

## 项目简介

SectorPulse 遵循“准二线策略”：通过市场情绪闸门决定是否开仓，在最强板块中挑选排名 3–5 的个股，并在不同状态下执行截然不同的操作。系统引入状态机和闭环流程，全面使用两融净偿还 + 涨跌停比 + 成交额集中度（Alt‑Flow）替代传统的北向资金口径，降低数据依赖与维护成本。

## 核心理念

  - 状态机驱动：根据市场情绪和账户仓位比率自动切换进攻、持仓、观望和休眠四种状态，每种状态具有明确的操作优先级。

  - 情绪闸门与 Kill‑Switch：在每日 10:00 和 14:00 执行情绪闸门检查，只有涨停家数、波动分位、指数位置、涨跌停比和成交额集中度全部满足阈值才允许开新仓；不满足则进入观望状态，触发 Kill‑Switch 时无条件清仓并休眠。

  - Alt‑Flow 替代口径：以两融净偿还（日终）为锚，结合盘中涨跌停比和成交额集中度判断市场资金情绪，并用波动分位保护形成风险分级。

  - 闭环决策：决策流程由账户状态开始，以复盘反馈结束，形成完整闭环，解决线性流程的漏洞。

## 核心功能
1. 状态机与风控
   
状态	| 市场情绪条件	     | 仓位条件	    |                   操作要点
进攻态	| 情绪闸门全部通过	 | 仓位 < 60%	| 在最强板块选股，单股首仓占总资产 10%，预设止损 −4%、止盈 +8%/+15%
持仓态	| 情绪闸门通过	     | 仓位 ≥ 60%	| 禁止开新仓，严格执行止盈/止损和时间衰减，板块退出 Top 3 时提前止盈
观望态	| 情绪闸门未通过	 | 仓位 ≤ 30%	| 绝对禁止新仓，按风险偏好对持仓执行更激进的止损/止盈
休眠态	| 触发 Kill‑Switch	| 任意仓位	    | 无条件清仓，暂停所有自动任务，须连续 3 个交易日情绪闸门通过且月度回撤修复至 −3% 以内方可重启

Kill‑Switch 分为 L1 （风险收缩）和 L2 （系统风险）两级：当两融净偿还或涨跌停比触发阈值或成交额集中度与指数下跌组合达到条件时，系统暂停新单或无条件清仓。

1. 数据服务与抽象层

数据层使用 AkShare 与东财等数据源采集市场数据，并在出现接口失败时自动降级到备用源，返回结构化的 FetchResult 并打标 degraded。
项目提供统一的数据抽象接口，支持获取指数日线、市场实时快照、涨跌停池、两融数据等，避免业务代码直连第三方库。
详细的 AkShare 接口及参数说明请参见 /docs/03. 数据接口清单（AkShare v1.17.40）.md。

3. 策略引擎与决策流程

策略引擎在每日 10:00 和 14:00 定时唤醒：先执行情绪闸门检查，若通过则根据账户仓位和迟滞逻辑判断当前状态，并分支执行“选板→选股→买点确认→下单意向”；否则进入观望/休眠流程，仅管理持仓。
持仓管理采用 T+3 观察、T+5 强制清仓及板块持续性监控，观望态则采取更激进的风控策略。

4. 持久化与 API

系统使用 SQLModel/SQLAlchemy 设计七张核心表，包括账户、持仓、交易、信号、意向、状态快照和事件日志，字段命名对齐 Alt‑Flow 口径。
持久化层使得系统崩溃后可恢复状态。
项目提供 FastAPI 服务用于内部通信和可选的微信指令接口，Streamlit 看板用于可视化当前状态、候选股和风控指标。

## 系统架构

项目采用分层架构：
   1. 数据服务层 – data_service/collector.py 和 providers/proxies 统一采集并清洗市场数据，提供 Alt‑Flow 聚合接口，并支持分钟数据降级。

   2. 策略引擎层 – strategy_engine/core.py 定义状态机及迟滞逻辑；gates/ 读取 Alt‑Flow、波动和趋势条件执行情绪闸门与 Kill‑Switch；strategies/sector_pulse_v3.py 实现选板、选股和买点确认。

   3. API 服务层 – api/main.py 基于 FastAPI 提供状态查询、信号触发和可选的微信接口。

   4. 持久化层 – models/database.py 定义数据库模型；pydantic_models.py 用于序列化和校验数据结构。

   5. 交互层 – web_dashboard/app.py 提供最小可视化仪表盘；wechat_service.py 实现微信消息推送（可选）。

   6. 调度与工具 – utils/scheduler.py 使用 APScheduler 在 Asia/Shanghai 时区内只在 10:00 和 14:00 触发任务，内置交易日历缓存和降级机制；
   其他工具包括 logger、helpers 等。

## 目录结构

- 项目目录遵循 v1.2 文档规定的规范命名：

SectorPulse/
├── api/                  # FastAPI 服务及路由
├── config/               # YAML 配置模板（Alt‑Flow 阈值、时窗/时区等）
├── data_service/         # 数据抽象层：collector.py、providers/、proxies/
├── docs/                 # 蓝图、开发文档、开发计划及接口清单
├── models/               # 数据库模型和 Pydantic 模型
├── services/             # 业务服务层：账户、持仓、微信服务
├── strategy_engine/      # 状态机核心与策略逻辑
├── utils/                # 调度器、日志与辅助工具
├── web_dashboard/        # Streamlit 可视化界面
├── main.py               # 入口，启动调度器:contentReference[oaicite:27]{index=27}
└── requirements.txt      # 依赖清单:contentReference[oaicite:28]{index=28}

## 安装与运行
  - 环境准备
      推荐使用 Python 3.10 或更高版本，确保时区设置为 Asia/Shanghai。
      安装依赖：项目使用 AkShare、APScheduler、FastAPI、SQLModel、Streamlit 等库。

  - 克隆仓库
      git clone https://github.com/EvanLee85/Sector_Pulse.git
      cd Sector_Pulse

  - 创建虚拟环境并安装依赖
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt

  - 初始化数据库（自动创建七张表）
  python -m main


默认情况下运行 python -m main 将启动调度器并维持空循环；定时任务将在每日 10:00 和 14:00 触发策略主循环。
使用 CTRL+C 退出时，调度器会安全关闭。

## 配置调整

  - config/base.yaml 和 config/strategy_core.yaml 提供默认阈值，包括进入/退出进攻态的仓位比率、Kill‑Switch 分级阈值、情绪闸门条件等。

  - 时间窗和时区设置在 config/base.yaml 的 execution.windows 和 tz 字段，可以根据需要调整，例如修改触发时间或更改到其他时区。

  - 数据采集降级策略可在 config 中配置，允许标记分钟数据缺失时的应对方式，例如 allow_fallback、on_missing 等。

## 风险与迭代规划

项目在 v1.2 版本完成了 Alt‑Flow 迁移，未来迭代计划包括：

  - 微信集成：提供信号推送与指令闭环。

  - 进阶复盘：在可视化界面中支持基于 EventLog 和 StateSnapshot 的高级复盘页。

  - 可选参数化：支持配置尾盘“仅卖出”时间窗等高级特性。

开发计划文档提供了逐步推进的任务清单，从环境搭建、数据库初始化、情绪闸门和 Kill‑Switch 实现、状态机开发、选板选股、买点确认、下单意向、持仓管理到实盘模拟及可视化等，每一步均列出目标、产出、验收标准和风险回退策略。
建议根据个人时间灵活推进，并在完成每个步骤时参考文档中的验收标准进行自查。

## 数据接口

项目大量使用 AkShare 数据源，但通过数据服务层提供统一入口，禁止业务逻辑直接调用 AkShare。数据接口清单文档列出了所有使用的 AkShare API，包括涨停股池、昨日涨停股池等，并给出参数说明、返回字段和示例。
如果接口失败，系统会自动降级到备用源（如新浪、雪球），并在返回结构中标注 degraded=True 与 note。

## 贡献和许可证

本仓库尚处于个人开发阶段，欢迎提交 issue 或 pull request 讨论改进建议。
请在提交代码前阅读开发文档和目录规范，确保文件命名与结构对齐。项目代码遵循 MIT 许可协议，依赖库版本已在 requirements.txt 中锁定并遵循原许可。
启动或发布该系统仅供学习和研究之用，不构成任何投资建议。