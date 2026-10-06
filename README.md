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

求解结果是**三态**证据，服务严格依据证据裁决，绝不把「未判定」说成「无解」：

| 证据 | 结论 | HTTP |
| --- | --- | --- |
| 找到可行解（含首次求解虽 `UNKNOWN` 但已持有完整可行 incumbent） | `feasible=true`（含合法时间轴与地址）；未获全部最优证明时 `optimal=false` | 200 |
| 首次求解**证明** `INFEASIBLE`，或放宽探针证明某资源不可能满足 | `feasible=false` + 已证 `reason` | 200 |
| 首次求解 `UNKNOWN`、无可用 incumbent，且没有任何探针给出不可能性证明 | `feasible=false, reason="undetermined"` | **503** |

两个诊断探针分别**放宽显存**（显存提到必然放得下的上界、仍受原 deadline
约束；该探针若被证无解，说明给多少显存都赶不上 deadline，原因才是
`deadline`）与**放宽截止时间**（截止时间提到必然排得下的上界、仍受原
memory_limit 约束；该探针若被证无解，说明排多久都放不下，原因才是
`memory_limit`）。在「放宽后的」问题上被证明不可能，才允许把该资源记为
无解原因；探针自身若返回 `UNKNOWN`，则该探针不提供任何证明，绝不能据此
臆测精确的资源原因。若首次求解本身已证无解、但探针无法锁定具体资源，仍
返回 200 且 `reason=null`（确属无解，但不臆造原因）。

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
- 返回：
  - 可行：HTTP **200**，`feasible=true`，含时间轴 `timeline`（每条路径
    起止、各算子起止、该路径 `peak_bytes`）与所有张量地址 `addresses`；
  - 已证无解：HTTP **200**，`feasible=false`，`reason` 为 `deadline` /
    `memory_limit` / `deadline_and_memory_limit`（仅在对应资源被探针
    **证明**不可能时给出；确属无解但无法锁定单一资源时 `reason=null`）；
  - 求解超时未判定：HTTP **503**，`reason="undetermined"`。这表示 CP-SAT
    在时限内既没找到可行解、也没有证明无解，**不是**「无解」，可稍后重试；
  - 任何结构或语义错误返回 **HTTP 422**。
- 可选请求头 `X-Solver-Timeout-Seconds: <正数秒>`：覆盖每次 CP-SAT 求解
  的时间预算（首次求解、各字典序阶段、两个诊断探针统一使用该值），用于
  稳定复现 / 核查超时行为。非法值（非数字、非正、NaN、无穷）返回 422。

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

### 无解示例

```bash
# P2 单独也需要 7，截止 6 无解
curl -s -X POST http://localhost:8000/plans -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json;d=json.load(open("examples/request.json"));d["deadline"]=6;print(json.dumps(d))')"
# {"feasible":false,"optimal":false,"reason":"deadline",
#  "capacity_upper_bound":0,"timeline":[],"addresses":[]}

# 最小容量为 96，显存给 95 无解
curl -s -X POST http://localhost:8000/plans -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json;d=json.load(open("examples/request.json"));d["memory_limit"]=95;print(json.dumps(d))')"
# {"feasible":false,"optimal":false,"reason":"memory_limit", ...}
```

### 求解超时未判定示例（HTTP 503，可复核）

用大规模输入 [`examples/stress_payload.json`](examples/stress_payload.json)
配合极小的求解时间预算，即可稳定让 CP-SAT 首次求解与两个探针都超时
（`UNKNOWN`）。该实例在默认预算下其实可行（1~2 秒内返回
`feasible=true`），因此这里的 503 确实是「超时未判定」而非无解——修复前
这种情况会被误报成 `feasible=false` 且臆造一个 `reason`：

```bash
curl -s -i -X POST http://localhost:8000/plans \
  -H 'Content-Type: application/json' \
  -H 'X-Solver-Timeout-Seconds: 0.00001' \
  -d @examples/stress_payload.json
# HTTP/1.1 503 Service Unavailable
# ...
# {"feasible":false,"optimal":false,"reason":"undetermined",
#  "capacity_upper_bound":0,"timeline":[],"addresses":[]}
```

同一个请求去掉该请求头（或给足时间）即返回 200 与可行规划，可对照复核。
超时请求头取值非法时返回 422：

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/plans \
  -H 'Content-Type: application/json' -H 'X-Solver-Timeout-Seconds: abc' \
  -d @examples/request.json
# 422
```

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
  热启动。任一后续阶段只拿到可行解而未获最优证明、或阶段超时，都会保留
  当前合法解并把 `optimal` 置为 `false`。
- 首次求解与两个诊断探针都按**可行 / 已证无解 / 未判定（UNKNOWN）**三态
  处理：仅在放宽后的问题上被证明不可能才报告对应 `deadline` /
  `memory_limit` / 二者原因；首次求解未判定且探针拿不到不可能性证明时，
  返回 `reason="undetermined"`（HTTP 503），绝不冒充已证无解，也不凭
  UNKNOWN 探针臆测精确资源原因。

---

## 4. 测试

```bash
pip install httpx
python3 tests/test_planner.py   # 21 项规划语义/最优性/三态裁决测试
python3 tests/test_api.py       # 11 项 HTTP 契约测试（含 503 超时与示例比对）
```

测试覆盖：互斥路径时间轴重叠与地址复用、同路径同时存活张量仍互斥、并行
fork-join 共存、半开区间边界复用、视图延长基张量占用、跨路径复用、
截止 / 显存无解分类、**首次求解或探针 UNKNOWN 时的三态裁决（不冒充无解、
不臆造资源原因、已证探针仍可锁定资源、后续阶段超时保留合法 incumbent）**、
**真实 CP-SAT 超时经 `X-Solver-Timeout-Seconds` 返回 503**、同名张量
一致性、依赖环、视图越界、422 契约，以及大规模实例的性能与“不轻易声称
最优”。

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
