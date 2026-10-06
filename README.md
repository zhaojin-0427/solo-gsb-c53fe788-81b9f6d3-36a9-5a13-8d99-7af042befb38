# 推理图缓冲区规划服务（Inference Graph Buffer Planner）

基于 **Python 3.11 + FastAPI + Google OR-Tools（CP-SAT）** 的 HTTP 服务。对多条
**互斥执行路径**联合求解：

1. 每条路径内是算子的无环依赖图，无依赖的算子可以并行；
2. 路径之间互斥（运行时只会执行其中一条），因此**不做串行排布**：每条路径在
   共享时间轴上独立规划、各自满足 `deadline`，不同路径的时间轴允许重叠；
3. 张量按 2 的幂对齐放入一块共享显存；张量自生产者完成时刻起、到最后一个
   消费者完成时刻止占用其**半开**字节区间，最终输出张量保留到路径结束；
4. 视图张量（`base_tensor` + `view_offset`）借用基张量的字节，并把基张量的
   占用延长到该视图的最后一次消费 / 路径结束；
5. 只有**同一路径**中时间上同时存活的不同基张量才不得重叠；属于互斥路径的
   张量即使时间轴重叠，也可以复用同一地址（地址空间全局共享）。

优化目标严格按以下字典序：

1. **最小化地址空间上界**（`capacity_upper_bound`，即所需显存最小峰值）；
2. 在容量固定的前提下，按**张量 ID 序**依次最小化各基张量地址（视图随基
   张量移动，地址 = 基地址 + 累计偏移）；
3. 再按**路径 ID 序**依次最小化各路径起点；
4. 最后按 **（算子 ID，路径 ID）序**依次最小化算子起点。

`optimal=true` 仅当 CP-SAT 对以上每一阶段都证明了最优；只要任何一个阶段
在时限内未能证明，返回的仍是合法可行解，但 `optimal=false`。

求解证据始终按**三值**对待：

- **可行**（CP-SAT 返回 OPTIMAL/FEASIBLE）：返回合法的时间轴与地址；
- **已证无解**（INFEASIBLE）：返回无解结论，且仅当两次资源诊断探针都有
  明确结论时才标注具体 `reason`；
- **未判定**（UNKNOWN，通常是超时）：既不冒充可行也不冒充无解，也不臆测
  是 deadline 还是 memory_limit 造成的。

三种裁决在 HTTP 层同样可区分（见第 2 节）。

---

## 1. 一键启动（Docker Compose）

需要本机已安装 Docker 与 Docker Compose v2：

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`。健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

交互式 API 文档（Swagger UI）：<http://localhost:8000/docs>

停止：

```bash
docker compose down
```

如需复现求解超时（未判定，HTTP 503），可在宿主机用环境变量把各阶段 CP-SAT
时限压到 0 后再启动（compose 文件会透传这些变量）：

```bash
PLANNER_STAGE1_SECONDS=0 PLANNER_PROBE_SECONDS=0 docker compose up --build
```

默认值（10s / 0.3s / 0.08s / 5s）写在 `docker-compose.yml` 中，与代码内置
默认完全一致；不设置这些变量时行为不变。

### 不使用 Docker（本地开发）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 2. 访问 `POST /plans`

- `Content-Type: application/json`
- 请求体顶层：
  - `deadline`（正整数）：每条路径必须各自在此全局时刻之前（含）完成；
  - `memory_limit`（正整数）：可用显存上界（字节）；
  - `paths`：路径数组，每条路径含 `id`、`operators`、`tensors`。
- 算子：`id`、`duration`（正整数）、`depends_on`（同路径算子 ID，无环、无
  自依赖、无重复）。
- 张量：
  - `id`、`bytes`（正整数）、`alignment`（正整数，必须是 2 的幂）；
  - `producer`：唯一生产者算子；`consumers`：消费者算子列表（生产者不会
    出现在消费者中，引用必须存在）；
  - `output`：可选，`true` 表示最终输出，占用保持到路径结束；
  - `base_tensor` / `view_offset`：可选，表示该张量是基张量上的视图；
    累计偏移必须按自身对齐对齐，且视图区间不得超出基张量；视图链不得成环；
  - 同名张量在不同路径中的定义（字节数、对齐、基张量、偏移）必须完全一致。
- 返回（三种裁决，HTTP 层可区分）：
  - **可行 — HTTP 200**：`status="feasible"`、`feasible=true`，含时间轴
    `timeline`（每条路径起止、各算子起止、该路径 `peak_bytes`）与所有张量
    地址 `addresses`；
  - **已证无解 — HTTP 200**：`status="infeasible"`、`feasible=false`。
    仅当放宽显存、放宽截止时间两次诊断探针都拿到明确结论时，`reason` 才为
    `deadline` / `memory_limit` / `deadline_and_memory_limit`；若某支探针
    超时未判定，无解本身成立但**精确资源原因不可证**，此时 `reason=null`；
  - **未判定 — HTTP 503**：首次求解与两支探针都没能在时限内给出可行解或
    无解证明（CP-SAT 的 UNKNOWN）。`status="unknown"`、`feasible=null`、
    `reason=null`。这**不是**无解结论，请求可稍后重试；首次 UNKNOWN 绝不
    会被当作已证无解。
  - 特殊情况：首次求解 UNKNOWN，但某支放宽探针找到了**同时满足原始
    deadline 与 memory_limit** 的可行见证，则升级为 HTTP 200 可行响应
    （`optimal` 必为 `false`），返回该见证对应的合法时间轴与地址。
- 无论可行与否，`optimal=true` 都要求全部字典序阶段都获最优证明。
- 任何结构或语义错误返回 **HTTP 422**。

### curl 示例（与实现实际输出一致）

请求文件见 [`examples/request.json`](examples/request.json)：

```bash
curl -s -X POST http://localhost:8000/plans \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

实际响应（已保存在 [`examples/response.json`](examples/response.json)，测试会
逐字段比对）：

```json
{
  "status": "feasible",
  "feasible": true,
  "optimal": true,
  "reason": null,
  "capacity_upper_bound": 96,
  "timeline": [
    {
      "path_id": "P1",
      "start": 0,
      "end": 5,
      "peak_bytes": 64,
      "operators": [
        {"operator_id": "a", "start": 0, "end": 2},
        {"operator_id": "b", "start": 2, "end": 5}
      ]
    },
    {
      "path_id": "P2",
      "start": 0,
      "end": 7,
      "peak_bytes": 96,
      "operators": [
        {"operator_id": "c", "start": 0, "end": 4},
        {"operator_id": "d", "start": 4, "end": 7}
      ]
    }
  ],
  "addresses": [
    {"tensor_id": "w",  "address": 64, "bytes": 32},
    {"tensor_id": "x",  "address": 0,  "bytes": 32},
    {"tensor_id": "y",  "address": 0,  "bytes": 64},
    {"tensor_id": "yv", "address": 16, "bytes": 16}
  ]
}
```

这个例子展示了全部关键行为：

- `P1`、`P2` 互斥但**不串行**：二者都从时刻 0 开始，各自独立满足
  `deadline=8`（若强制串行，总时长 12 将超限而被误判为无解）；
- `x` 是 `P1` 的最终输出，占地址 `0..32` 到路径结束；`w` 在 `P1` 中被
  `b` 消费、与 `x` 同时存活，放到地址 64（张量 ID 序下的字典序最小合法
  布局，`P1` 峰值 64）；
- `y` 在 `P2` 中从 `c` 完成（时刻 4）存活，其输出视图 `yv` 把它延长到
  路径结束（时刻 7）；`w` 在 `P2` 同样存活到 7，因此二者必须错开，
  `P2` 峰值与全局容量上界均为 **96**；
- `yv` 借用 `y` 的字节：地址 = 0 + 16 = 16；
- `y` 与 `P1` 时期的 `x` 分属互斥路径，即使时间轴完全重叠也都能使用
  地址 0（跨路径复用）；同名张量 `w` 在两条路径中共享同一全局地址 64。

### 已证无解示例（HTTP 200，`status="infeasible"`）

```bash
# P2 单独也需要 7，截止 6 无解
curl -s -X POST http://localhost:8000/plans -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json;d=json.load(open("examples/request.json"));d["deadline"]=6;print(json.dumps(d))')"
# {"status":"infeasible","feasible":false,"optimal":false,"reason":"deadline",
#  "capacity_upper_bound":0,"timeline":[],"addresses":[]}

# 最小容量为 96，显存给 95 无解
curl -s -X POST http://localhost:8000/plans -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json;d=json.load(open("examples/request.json"));d["memory_limit"]=95;print(json.dumps(d))')"
# {"status":"infeasible","feasible":false,"optimal":false,"reason":"memory_limit",
#  "capacity_upper_bound":0,"timeline":[],"addresses":[]}
```

### 超时未判定示例（HTTP 503，`status="unknown"`）

先以零求解时限启动服务（见第 1 节）：

```bash
PLANNER_STAGE1_SECONDS=0 PLANNER_PROBE_SECONDS=0 docker compose up --build
```

再发送同一个请求：

```bash
curl -i -s -X POST http://localhost:8000/plans \
  -H 'Content-Type: application/json' -d @examples/request.json
# HTTP/1.1 503 Service Unavailable
# ...
# {"status":"unknown","feasible":null,"optimal":false,"reason":null,
#  "capacity_upper_bound":0,"timeline":[],"addresses":[]}
```

含义是：时限内**既没找到可行解、也没证明无解**。它不携带 `reason`，不能
当作无解结论；恢复默认时限（`docker compose up --build`）重新请求即可得到
上面的 200 可行响应。若 CP-SAT 很快证明无解、但某次诊断探针超时，响应仍是
200 + `status="infeasible"`，只是 `reason` 为 `null`（无解可证、具体资源
原因不可证）。

### 无效输入（HTTP 422）

- 非正整数耗时 / 字节数 / 截止时间 / 显存、对齐不是 2 的幂、未知引用、
  重复 ID、依赖环、视图越界或成环、跨路径同名张量定义不一致、多余字段、
  JSON 语法错误等，均返回 422：

```bash
curl -i -X POST http://localhost:8000/plans -H 'Content-Type: application/json' \
  -d '{"deadline":-1,"memory_limit":1024,"paths":[]}'
# HTTP/1.1 422 Unprocessable Content
```

---

## 3. 建模要点

- 每个算子一个非抢占区间变量，`depends_on` 与生产者→消费者（及基张量
  生产者→视图生产者）都建模为先序约束；无依赖算子通过 CP-SAT 的
  `NoOverlap` 之外的自由排布实现并行。
- 路径窗口由各算子起止的 min/max 决定，每条路径独立受 `deadline` 约束；
  互斥路径之间**没有** `NoOverlap`，时间轴可以重叠。
- 内存放置使用**按路径分别施加**的二维条带装箱 `AddNoOverlap2D`：每个
  「路径 × 基张量」的存活期是一个 `(时间区间, 字节区间)` 矩形，所有路径
  共享同一组地址变量（全局显存）。该约束一次性保证：同一路径同时存活的
  不同基张量字节不重叠；不同路径的矩形从不出现在同一约束中，因此互斥
  路径即使时间轴重叠也能复用同一字节；`地址 + 大小 <= capacity <=
  memory_limit` 保证不超显存。
- 地址使用模约束按 2 的幂对齐；视图链取链上最严格的对齐放置基张量。
- 求解采用单模型增量字典序：先最小化容量并固定，再逐个最小化张量地址、
  路径起点、算子起点，每个变量优化后立即固定；每个阶段带解提示（hint）
  热启动，已找到的可行现任解会快照保留——后续阶段即使超时也仍返回该合法
  解，只是 `optimal=false`。
- 可行性诊断分别放松截止时间与显存各求解一次。**三次求解（首次 + 两支
  探针）的 OPTIMAL/FEASIBLE、INFEASIBLE、UNKNOWN 证据分别记录**：探针
  UNKNOWN 时绝不按“不可行”处理；只有两支探针都确定时才给精确 `reason`，
  首次 UNKNOWN 且无可行见证时整体裁决为未判定（HTTP 503）。

---

## 4. 测试

```bash
pip install httpx
python3 tests/test_planner.py   # 24 项规划语义/最优性/三值证据裁决测试
python3 tests/test_api.py       # 10 项 HTTP 契约测试（含示例响应逐字段比对）
```

测试覆盖：互斥路径时间轴重叠与地址复用、同路径同时存活张量仍互斥、并行
fork-join 共存、半开区间边界复用、视图延长基张量占用、跨路径复用、
截止 / 显存无解分类、同名张量一致性、依赖环、视图越界、422 契约，以及
大规模实例的性能与“不轻易声称最优”。三值证据部分覆盖：首次求解超时
（UNKNOWN）上报 HTTP 503 `unknown` 而非误报无解、UNKNOWN 被探针可行见证
救回为合法非最优计划、首次 INFEASIBLE 但探针超时时 `reason=null`、单支
探针 UNKNOWN 不臆测资源原因、后续字典序阶段超时时保留合法现任解。

## 目录结构

```
app/
  main.py       # FastAPI 路由、422 异常映射
  schemas.py    # Pydantic 请求/响应模型（结构校验）
  planner.py    # 语义校验 + CP-SAT 建模 + 字典序求解
examples/
  request.json          # 文档中的真实请求
  response.json         # 该请求当前实现的真实响应
  stress_payload.json   # 大规模测试输入
tests/                  # 语义与 HTTP 测试
Dockerfile
docker-compose.yml
requirements.txt
```
