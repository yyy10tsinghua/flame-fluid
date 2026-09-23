# Flame Fluid — 实时二维流体火焰模拟

基于 **Navier-Stokes 流体求解** 的二维火焰：分叉、摇曳、飘动、渐隐消失、半透明，强度与范围实时可调，火星为随流体场飘动的粒子系统（随机生成 / 熄灭 / 爆裂弹射）。

![界面总览](docs/images/ui-overview.png)

**在线预览**：本目录起静态服务器（如 `python -m http.server 4173`），浏览器打开 `flame-fluid/index.html`。所有参数面板实时生效，可导出透明序列帧。

---

## 一、五种预设效果

| 篝火 bonfire（默认） | 蜡烛 candle |
|---|---|
| ![篝火](docs/images/preset-bonfire.png) | ![蜡烛](docs/images/preset-candle.png) |
| 中等火势，分叉明显，带烟与火星 | 小火苗、强蓝焰底、微风即晃、几乎无烟 |

| 火把 torch | 壁炉 hearth | 烈焰 blaze |
|---|---|---|
| ![火把](docs/images/preset-torch.png) | ![壁炉](docs/images/preset-hearth.png) | ![烈焰](docs/images/preset-blaze.png) |
| 窄火源、高湍流、火星多 | 宽火床、低浮力、浓烟 | 大范围高强度、火星爆裂密集 |

以上均为**直通 alpha 透明 PNG**（384×576），可直接叠在任何游戏背景上。

---

## 二、核心技术点

### 2.1 稳定流体求解（Stam Stable Fluids）

火焰的运动不是噪声动画，而是真实求解不可压缩流体：

- **半拉格朗日对流**：速度场自我输运时，每个网格点沿速度回溯采样
  `x_back = x − v·dt`，无条件稳定，任意大时间步不爆炸。速度与所有标量场（温度/燃料/烟）都用这条通路。
- **压力投影（Jacobi 迭代）**：对速度场求散度后，迭代解泊松方程
  `∇²p = ∇·v`，再减去压力梯度 `v ← v − ∇p`，得到无散度场——这是流体"不可压缩"的数学表达，火焰翻卷时体积守恒、出现真实涡旋。
  实现：26 次迭代 + 上一帧压力**热启动**（收敛快一倍），WebGL 上就是 26 个全屏 quad pass。
- **边界处理**：
  - 底面：法向速度 ≥ 0（火源只出不进），切向滑移；
  - 顶部：**常开**（压力 Dirichlet `p=0`）——热气流可流出域外。这点很关键：封顶会导致热堆积、火柱被"天花板"压回再循环，视觉上剧烈脉动；
  - 侧壁：**封闭（滑移壁）/ 开放 可切换**。封闭时火苗被风吹到边上贴壁卷回（壁炉感）；开放时气流穿出画面（户外篝火、行进火把）。

| 侧壁封闭 + 强风：火焰贴壁堆积 | 侧壁开放 + 强风：火苗被吹出画面 |
|---|---|
| ![封闭](docs/images/boundary-closed.png) | ![开放](docs/images/boundary-open.png) |

### 2.2 燃烧模型（四通道标量场）

材质纹理一张 `RGBA16F` 同时存四个物理量：**R=温度 T，G=燃料 F，B=烟 S，A=余烬 E**，全部被速度场被动输运，每帧经历：

1. **火源注入**：底部火源带内注入温度与燃料。注入强度乘**双频 value 噪声**（空间 5.5 周期 + 时间流动），相邻火舌点火时序错开——这是**分叉**的来源：不同火舌各自随机点燃、熄灭、再点燃。
2. **燃烧反应**：`react = F · clamp((T−T_ign)·4) · burnRate · dt`，燃料遇点火温度放热并生烟。放热量调得很低（`heatRel=0.16`）且温度钳制在 1.45——放热过强会把整条火柱顶成饱和白光柱。
3. **边缘侵蚀**：温度乘噪声风化掩码（火源中心权重 + 双频噪声），只切边缘不伤主体——火舌分离、撕裂感。
4. **高度平方冷却**：`T ·= exp(−cooling · (0.45 + y²·2.0) · dt)`。冷却率随高度平方增强，火苗自然收尖；配合顶部渐隐实现"消失"。
5. **烟/火分离渐隐**：温度在 72%–98% 高度渐隐，烟在 82%–100% 渐隐——火熄了烟还在飘。

### 2.3 涡度约束（Vorticity Confinement）

数值求解会不断抹掉小旋涡（数值耗散），火焰会越烧越"糊"。涡度约束每帧计算涡量场 `ω = ∂v/∂x − ∂u/∂y`，向涡量集中处反向补一个力：

```
N = ∇|ω| / |∇|ω||          （涡量梯度方向）
f = ε · (N × ω)             （垂直于涡量梯度，推动旋转）
```

`ε` 即面板上的**湍流**参数（0–4）：0 是呆板的直柱，2.3（篝火默认）有清晰翻卷分叉，4 接近剧烈搅动。这是火焰"活"感的关键来源。

### 2.4 火星粒子系统（随流体飘动 + 爆裂）

火星不是屏幕空间噪声（那是"固定速度上移的亮斑"假象），而是**完整的 GPU 粒子系统**：

- **数据布局**：256 个粒子存在两张 16×16 `RGBA16F` ping-pong 纹理里——`part(pos.xy, vel.zw)` 与 `meta(age, maxLife, seed, burst)`，更新就是两个全屏 pass。
- **随流体飘动**：每帧每个粒子在自身位置**采样流体速度场**（就是火焰本身的流场），以 `flow·0.92 + 自身惯性` 移动。火苗摇曳时火星跟着拐弯、被卷进涡旋——因为它就在那股气流里。
- **随机生死**：每粒子随机寿命 0.7–2.2s，到期在火源区随机位置重生；未中签的粒子（由 sparks 参数控制激活率）当帧休眠、下帧重掷骰，实现零内存分配的密度控制。
- **爆裂**：约 30% 新生粒子触发——出生瞬间径向弹射初速（斜向上），辉光值 burst 从 1.0 按 `exp(−4.5·dt)` 衰减，渲染尺寸与亮度随 burst 增大。观感就是火星"炸开"。
- **渲染**：顶点着色器 `texelFetch` 读粒子纹理（无顶点缓冲），`gl.POINTS` + 径向高斯衰减片元，additive 混合到一张 spark 层，再叠进主合成与 bloom。

### 2.5 渲染与合成管线

```
mat(温度/燃料/烟/余烬) ──► SHADE: fireRamp 色带 + 白核 + 蓝焰 + 烟照明
spark 粒子层 ──────────► additive 叠加
bloom: 半分辨率降采样 → 双 pass 高斯模糊 → 辉光
composite: 背景(夜晚/棋盘/黑白) + 烟 + 火 + 火星 + 辉光 + 抖动去色带
export: 预乘合成 → 直通 alpha 转换 → 透明 PNG
```

细节：

- **fireRamp 五段色带**：暗红 → 深橙 → 橙 → 亮黄 → 黄白，映射 `heat = T/1.25`。
- **白热核心限制**：白核只出现在 `heat > 0.955 且高度 < 22%` 的根部小区域。早期版本白核占亮部 50–65%（整柱过曝白板），收紧后降到 ~12%。
- **烟的火光照明**：烟色 = 冷灰蓝 与 暖橙 按 `exp(−dist_to_fire·2.6)·0.85 + 本地热度·0.6` 混合——近火焰的烟被火光照成暖棕，远处冷灰。只按本地温度染色会让火苗上方的烟永远是冷色（那里 T≈0）。
- **直通 alpha 导出**：片元内做 预乘 → 直通 转换（`col /= alpha`），导出的 PNG 在任何合成器里直接可用。
- **抖动去色带**：输出加 ±1/190 随机抖动，消灭暗部渐变条带。

### 2.6 CPU 移植（FlameSolver.ts）与 Cocos 组件

`cocos/FlameSolver.ts` 是**零依赖纯 TypeScript** 的同物理求解器，供 Cocos Creator 等引擎内实时运行：

- 与 GPU 版共享同一套物理常数（`K` 表逐一对应）；
- 对流用**双线性插值**回溯（对齐 GPU 的 LINEAR 采样；最近邻采样数值扩散大，烟会糊满整域）；
- 全部 TypedArray（Float32Array ×14）+ 就地交换，无 GC 压力；
- 火星粒子为同逻辑的 256 元素数组池；
- `render()` 直接产出直通 alpha 的 `Uint8Array`，喂给 `Texture2D.uploadData` 即成动态纹理。

`cocos/FluidFlame.ts` 组件：节点挂 Sprite + 本组件，面板参数与 demo 同名同义，`gridWidth` 控制精度（手机 64 / 平板 96），附 `applyPreset()` / `igniteAtNormalized()` 交互接口。

---

## 三、每帧 Pass 流水线（GPU 版）

| # | Pass | 输入 → 输出 | 说明 |
|---|---|---|---|
| 1 | curl | vel → curl(R) | 涡量场 ω |
| 2 | forces | vel,curl,mat → vel' | 浮力·风·阵风·涡度约束·粘滞·边界 |
| 3 | advect | vel' → vel'' | 速度场自对流（半拉格朗日） |
| 4 | divergence | vel'' → div(R) | 求散度 |
| 5 | jacobi ×26 | p,div → p' | 压力迭代（热启动+开放边界） |
| 6 | subtract | vel'',p → vel''' | 减压力梯度，得无散场 |
| 7 | material | vel''',mat → mat' | 标量对流+注入+燃烧+侵蚀+冷却 |
| 8 | partMeta / partPos | part,meta,vel → ' | 火星粒子更新（2 pass） |
| 9 | points | part,meta → spark | gl.POINTS additive 渲染粒子层 |
| 10 | bloom + blur×2 | mat,spark → bloomA | 辉光 |
| 11 | composite / export | 全部 → 屏幕 / 透明PNG | 最终合成 |

全部为全屏三角形 quad，192×288 网格下 33 个 draw call，桌面 GPU 实测 **0.08ms/帧**。

---

## 四、参数表（demo 面板 / FluidFlame 组件同名）

| 参数 | 范围 | 物理意义 |
|---|---|---|
| intensity 强度 | 0.05–1.5 | 火源温度与亮度 |
| sourceWidth 范围 | 0.02–1.0 | 底部火源宽度（占域比例） |
| vorticity 湍流 | 0–4 | 涡度约束 ε：分叉/翻卷强度 |
| fuel 燃料 | 0.1–2 | 火苗高度与持续性 |
| cooling 冷却 | 0.2–4 | 越大火焰越短、消失越快 |
| buoyancy 浮力 | 0–2 | 热空气上升加速度 |
| wind 风力 | −2–2 | 横向恒定风（负=向左） |
| gust 阵风 | 0–2 | 随机摇曳幅度 |
| smoke 烟雾 | 0–2 | 燃烧生烟量 |
| blueBase 蓝焰 | 0–1 | 根部蓝色底焰（蜡烛/燃气感） |
| sparks 火星 | 0–1.5 | 粒子激活密度 |
| glow 辉光 | 0–1.5 | bloom 强度（仅 demo/序列帧） |
| openSide 侧边界 | 封闭/开放 | 封闭=贴壁卷回；开放=气流穿出（配合风力） |

---

## 五、移动端可行性（实测数据）

桌面基准（平板按单核 1/3–1/5 折算）：

| 方案 | 网格 | 桌面耗时 | 平板折算 | 帧率预期 |
|---|---|---|---|---|
| GPU 流体（着色器移植 Cocos effect） | 128–256 宽 | 0.04–0.08 ms | 0.4–1.2 ms | 60fps 无压力 |
| CPU 组件 FluidFlame | 48×72 | 1.7 ms | 5–9 ms | 60fps 可达 |
| CPU 组件 FluidFlame | 64×96 | 2.8 ms | 8–14 ms | 30fps 稳 / 60 边缘 |
| CPU 组件 FluidFlame | 96×144 | 5.9 ms | 18–30 ms | ≤30fps |
| 序列帧 Sprite | — | ≈0 | ≈0 | 任意 |

**选型建议**：装饰性火焰（参数固定）→ 序列帧最省心；需要动态性（风力变化、被点燃/浇灭）→ 普通平板 `gridWidth=64` 锁 30fps，可同屏 2–3 团；要 60fps 大火焰 → 把 demo 的 GLSL 移植成 Cocos 自定义 effect，GPU 跑流体（iOS 15+/近年安卓均支持所需浮点渲染目标）。

---

## 六、目录结构

```
flame-fluid/
├─ README.md               本文档
├─ index.html              WebGL2 demo: 单文件零依赖 (求解器+面板+粒子+导出)
├─ docs/images/            效果截图 (本文档引用)
├─ cocos/
│  ├─ FlameSolver.ts       纯 TS CPU 求解器 (无引擎依赖, 含火星粒子)
│  ├─ FluidFlame.ts        Cocos Creator 3.x 组件 (动态纹理实时火焰)
│  ├─ test-solver.mjs      Node 离线渲染验证 (可当批导出管线)
│  ├─ bench-solver.mjs     各网格档位 CPU 耗时基准
│  └─ solver-*.png         CPU 求解器输出样张 (5 预设)
├─ export/
│  └─ flame_sample_*.png   篝火稳态三连帧 (384×576 直通 alpha)
└─ tools/
   └─ analyze-png.mjs      帧像素分析 (火焰覆盖/颜色/半透明统计)
```

## 七、Cocos Creator 集成

**方式 A · 实时组件**（动态火焰、可交互）：

1. `FlameSolver.ts` + `FluidFlame.ts` 复制进 Cocos 项目脚本目录；
2. 场景节点挂 Sprite + FluidFlame，节点宽高即火焰显示尺寸；
3. 代码控制：

```ts
const flame = node.getComponent(FluidFlame)!;
flame.applyPreset("torch");               // bonfire/candle/torch/hearth/blaze
flame.intensity = 1.3; flame.wind = 0.8;  // 运行时实时生效
flame.igniteAtNormalized(0.5, 0.1, 1.2);  // 火焰箭落点点火
```

**方式 B · 透明序列帧**（装饰火焰、零开销）：demo 面板调参 → 「导出 48 帧序列 PNG」→ Cocos 建 SpriteFrame 序列 + AnimationClip 循环。循环跳变用两份错相动画交叉淡入消除。

## 八、已知边界

- CPU 组件版无 bloom 辉光（视觉略"干"），可挂引擎 Bloom 后处理或叠径向渐变光晕图；
- 序列帧导出保持页面前台（浏览器后台节流；demo 带 watchdog 以 ~10fps 兜底继续仿真）；
- 早期调参的关键教训均已固化在系数里：封顶边界会热堆积、放热过强成白光柱、最近邻采样烟糊满域、烟需按火源距离照明。
