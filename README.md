# BEP_MultiFlow / code

在 Robbins, Griffiths & Fenton (2021) 的基础上做两相流扩展的起步代码。全部是 PyTorch，
目标硬件是单块 RTX 5090；没有 MPI。

```
code/
  bepflow/
    darcy_pipe.py    Robbins 式稳态 Darcy + 管流模型，B 个蒙特卡洛样本一起推进
    pipe_el2d.py     二维 (x–z) 管段欧拉–拉格朗日求解器（NTLP 核心算法的 torch 重写）
  run_baseline.py    跑 darcy_pipe，输出 (H/L)cr 和图
  run_pipe.py        跑 pipe_el2d，输出图和每秒粒子步数
  verify.py          21 项检查（与解析解、与论文量级的对比）
  plotstyle.py       画图样式
  NTLP_notes.md      NTLP 学习笔记，含与 pipe_el2d.py 的逐例程对照
tmp/                 所有运行输出、日志、图；以及 torch_shim/（见“验证状态”）
```

## 运行

在项目根目录（`BEP_MultiFlow`）下：

```
python code\verify.py                     先跑这个；全部 PASS 再往下
python code\run_baseline.py               均质土，单个样本
python code\run_baseline.py --B 256 --cov-k 1.0 --theta 0.5 --tag mc256
python code\run_pipe.py                   2 万颗粒
python code\run_pipe.py --n 2000000 --float32 --tag big
```

脚本自动选 `cuda`，加 `--cpu` 强制用 CPU。输出写到 `tmp\`。


| 检查 | 结果 |
|---|---|
| 均质土水头梯度、流量 | 相对误差 1e-11 |
| 管道单元流量 = 立方定律，壁面剪应力 = ρg a i/2 | 1e-12 |
| 对数正态场均值、变异系数、Markov 相关、互相关 | 0.1%–0.6%（400 个样本的统计误差） |
| Poiseuille 平均流速、床面剪应力 | 3e-6 |
| 投影后散度 | 机器精度 |
| 颗粒沉速（Schiller–Naumann） | 5e-15 |
| 颗粒沉积长度（含一阶惯性修正） | 0.13% |
| 双向耦合动量守恒（x、z） | 机器精度 |
| 悬浮颗粒把水下重量传给流体 | 5e-12 |
| 均质土管道推进 (H/L)cr | 0.204；论文近均质工况约 0.17–0.20 |

## darcy_pipe.py

算法流程按论文图（先对管深做 Picard 迭代，再判断推进，再升水头），但：

1. 单元中心有限体积 + 调和平均，不是四节点有限元。
2. 随机场用 FFT 循环嵌入生成，取单元中心点值；论文用 LAS，带局部平均的方差折减。
   相关长度只有几个单元时两者方差略有差别。
3. 单元内水力梯度由按导水系数加权的面水头求得。与论文一样，它定义在一个单元的尺度上，
   所以临界梯度 i_cr 依赖网格尺寸（论文说明 0.3 对应 0.125 m 的单元）。
4. 上游水头按几何级数升高（每次 ×1.01）。论文的表述是“1 percent of the assigned value”，
   几何级数与论文说的每次分析 200–400 个计算步吻合。
5. 管道单元的导水系数直接取代土体单元（平面模型不允许水从管道下方绕过），这一点与论文相同，
   也是论文自己列出的第一条局限。

求解器是批量 Jacobi 预条件共轭梯度。它够用但不快：高变异系数时每次求解要几百次迭代。
要做论文那样 700 个工况 × 4400 个样本的规模，先换成多重网格预条件再说。

## pipe_el2d.py

验证的是“NTLP 那套算法能在单 GPU 上用 torch 张量写出来，并且数值正确”：

- 点粒子假设要求粒径小于网格间距、颗粒相稀疏。真实侵蚀管道深度只有几到几十个 d50，
  输沙以贴床的推移质为主，颗粒间接触不可忽略——都在这个假设之外。
  默认参数（管深 3 mm、粒径 0.1 mm、24 层）刚好卡在 d_p/dz = 0.8。
- 没有颗粒接触，没有床面再起动：颗粒落到床面就粘住。
- 二维流动不会转捩。管深超过约 4 mm（d50 = 0.2 mm、τ = τc 时 Re ≈ 1800）以后真实流动
  可能不再是层流，要三维计算。
- x 方向周期，所以没有来自床面的渗流入流（那需要进出口边界）。


|---|---|
| `sensitivity_pipe_factor.py` + `sensitivity_summary.py` | 同一批随机场下把管道导水系数乘 F，比较 (H/L)cr 和管道路径 |
| `sensitivity_tau_c.py` | 均质土，改 τ_c 和 i_cr |
| `sensitivity_mesh.py` | 均质土，40/80/160 网格；固定长度管道前方单元的梯度随网格的变化 |
| `tip_asymptotics.py` | 把一条直管道处处保持在 τ = τ_c，拟合管尖附近 a(r)、h(r) 的幂次 |
| `sensitivity_results.txt` | 以上全部输出的汇总 |

结论（均为 NumPy 替身上的结果）：这个模型的临界梯度几乎只由管尖判据 i_cr 决定
（(H/L)cr ≈ 0.68 i_cr），对管道导水系数（256 倍范围内变化不到 ±8%）和 τ_c（16 倍范围内不变）都不敏感；
管道自己把深度调到 a ∝ F^(-1/2) 来维持 τ = τ_c。阈值管道在管尖附近有 a ∝ r^(1/(2m))，
m 是阈值状态下 Q ∝ a^m 的指数（层流 2，粗糙湍流 1）；数值拟合得到 0.234（预测 0.25）和 0.480（预测 0.5）。


| 文件 | 内容 |
|---|---|
| `pipetheory/section.py` | 带刚性顶板的层流阈值断面：浅水极限公式、断面内 Stokes 流求解（对称断面和任意形状断面）、自由边界完整解（可带均匀的床面渗流 `sigma`） |
| `pipetheory/pipelaw.py` | 坡降–流量关系：自成形断面；管宽由管尖决定（`W0`）；弯曲度 |
| `pipetheory/aquifer.py` | 箱体内三维 Darcy 流和响应矩阵。孔口出流或堤脚出流、各向异性、出口透镜；余弦变换预条件的共轭梯度；`prescribed_pipe`：管内水头已知时的一次求解（管道可以贯通到上游面） |
| `pipetheory/coupled.py` | 管道与含水层耦合：折叠点、两个分支的伪弧长延拓、一般管道律的有量纲求解、管尖坡降、砂性参数 |
| `pipetheory/poldata.py` | 读 Pol (2022) 4TU 数据集的观测表和孔压记录，提取"管道静止"状态 |
| `run_section.py` | 断面系数随摩擦系数的表，与 Pol 2022 图 3.10 的实测断面对比 |
| `run_bifurcation.py` | 分岔图（两个分支）、沿管轴的水头、判据分区图 |
| `run_pol_box.py` | Pol 2022 小尺度试验的临界水头：8 种模型组合逐一对比（约 4 分钟） |
| `compare_pol_heads_all.py` | 全部静止状态：折叠水头与实际水头、流量、水头剖面（需要 `Data/Pol_4TU`；约 1.5 分钟） |
| `pipe_width_data.py` | 只用实测流量和管内坡降反推各试验的管深、过流宽度、管床渗流坡降（Pol 2022 与 Van Beek 2015 表 5；小、中尺度） |
| `run_scales.py` | 中尺度平衡状态和临界水头，四种管宽闭合对比（1 cm 网格，约 10 分钟） |
| `run_tip_criterion.py` | 各静止状态的管尖受力度量（入流强度、2 cm 割线坡降、平方根强度），小、中尺度对比（约 8 分钟） |
| `run_tip_same_sand.py` | 同一种砂（Baskarp）两个尺度：小尺度标定管尖强度，预测中尺度静止水头和临界水头，并反向检验（约 8 分钟） |
| `run_large_scale.py` | Pol 2022 第 4 章大尺度试验（与小尺度 FPH 同一种砂，20 倍尺度）：用小尺度标定预测管内坡降、所需水头、管尖受力（2.5 cm 网格，约 5 分钟） |
| `run_tip_nonlocal.py` | 筛选：管尖前方坡降取多长的距离才与尺度无关（固定 5 cm、砂层厚度的分数、渗径的分数），三个尺度（约 9 分钟） |
| `run_large_scale_width.py` | 大尺度：管道排水宽度（5 cm 到全槽宽）对管尖受力和所需水头的影响；`--taper` 让每个断面取管道律要求的宽度并迭代（每个算例约 0.5–1.5 分钟） |
| `make_figure_large_scale.py` | 稿件图 6（大尺度：管内水头；所需水头随排水宽度），读 `tmp/` 里上面两个脚本的输出 |
| `make_figure_regimes.py` | 稿件的三个区图：全部实测状态在无量纲坡降–流量图上，读 `tmp/paper1_pipe_width_data.csv` |
| `make_figures_extra.py` | 稿件的集总模型图、尺度效应总图（$H_c/L$ 对 $L$）和图形摘要 |
| `make_figure_plan.py`、`make_figure_section_data.py` | 稿件图 1(c)（平面示意）和图 2(c)（Pol 实测断面与阈值断面） |
| `make_figure_c3.py` | 两种出口几何（顶板开孔、坡脚出口）的尺度律系数随管长的变化（约 1 分钟） |
| `depth_profile_check.py` | 管深沿程分布：理论（自成形、管尖宽度律）对 Pol 式 3.1 的实测回归，不拟合（需要 `tmp/paper1_pol_states.csv`） |
| `large_scale_data.py` | 大尺度试验的原始记录（4TU 数据集，放在 `Data/data`）：试验过程、管内坡降、管尖前方坡降、进水端损失、由出砂量反推管道深度和宽度；不用模型 |
| `tip_profile_data.py` | 小尺度实测：以管尖为原点的平均水头剖面和坡降，不用模型（需要 `Data/Pol_4TU`） |
| `progression_rate_data.py` | 各试验推进阶段的速度、管内坡降和等效管深；检验"推进速度 × 管深"是否为砂的常数（需要 `Data/Pol_4TU`） |
| `pipetheory/laserdata.py` | 读 Pol (2022) 数据集里的原始激光记录（`pressure/*_laser.MAT` + `laser/` 里的时间窗）：横向对齐、去掉偏小的读数、唇口连线作顶板、mA 换算成深度 |
| `laser_sections.py` | 全部扫描的全部断面：床面剖面、最深通道的形状、侵蚀区宽度和面积、质量标记；检查图写到 `tmp/laser_sheets/`（约 15 秒；不用模型） |
| `laser_vs_theory.py` | 逐断面对比：扫描时刻的水头和流量、实测断面上的层流（最大床面剪应力、流量）、同坡降下的阈值断面、由实测管内水头算出的入流和床面渗流坡降（约 1 分钟；先跑 `laser_sections.py`） |
| `make_figure_laser.py` | 稿件图 `fig10_laser_sections`（断面形状、静止时的剪应力、生长中管道的管深、管深对流量即管道律的直接检验），数字写到 `tmp/paper1_laser_figure.txt`；同时把四个分图画的数据写到 `tmp/paper1_laser_figdata.json`，供 `make_figures_jfm.py` 的图 5 使用 |
| `run_section_seepage.py` | 含床面渗流的阈值断面（稿件表 2）：不同 σ 下的断面、流量系数和同流量下的坡降、管深、宽度 |
| `pipetheory/seepage.py` | 含床面渗流的管道：断面族（σ = 0…0.9，缓存在 `data/section_seepage_mu*.csv`）、局部律（σ 由当地流量和入流决定）、与砂层耦合的求解和折叠点；可选的惯性项 |
| `run_seepage_coupled.py` | 含渗流的耦合问题（稿件表 5）：逐断面预测和 29 个静止状态，对渗流力份额 ε = 0…1（约 6 分钟；先跑 `laser_vs_theory.py` 和 `compare_pol_heads_all.py`） |
| `run_section_convergence.py` | 阈值断面数值解的网格收敛、外推值和不同初值的试验（稿件附录 A；约 6 分钟） |
| `run_assumptions.py` | 29 个静止状态下各假设的量级：雷诺数、缓变、床面入流、滑移、管深与粒径之比、保持层流的管深（稿件 7.3 节；约 1 分钟） |
| `pipetheory/inertia.py` | 弱惯性（稿件第 4 节、附录 B）：宽管极限一阶解的精确系数（有理数运算）、宽的阈值管道、单壁入渗的 Navier–Stokes 相似解（核对用）、自成形断面的闭合系数 c、浅水极限下自成形管道的一阶解（`shallow_self_formed`，入渗分布可选）、岸坡与顶板交角处入渗的奇异指数 |
| `run_inertia.py` | 惯性（稿件第 4 节和 5.2 节）：宽管系数和相似解的对照表；浅水极限下精确解与闭合式之比随入渗分布的变化；29 个静止状态不含 / 含惯性项 / 惯性项乘浅水极限因子（约 1.21）的折叠水头、水头升、流量；折叠水头随 J0 = c k H0/ν 的曲线（约 12 分钟） |
| `make_figure_inertia.py` | 稿件图 `fig11_inertia`（先跑 `run_inertia.py`） |
| `make_paper_figures.py` | 旧版稿件用图（第二十四轮及以前的 `fig1_sketch`、`fig2_section`、`fig3_fold`、`fig4_small_scale`、`fig5_scales`）；稿件已不再引用，保留作对照 |
| `run_okamura.py` | Okamura 和 Kusube (2025) 模型堤几何下的推进坡降，与其预测和实测对比 |
| `compare_pol_heads.py` | 上一轮的单试验脚本（B25_245），已被 `compare_pol_heads_all.py` 取代，保留备查 |
| `verify_paper1.py` | 80 项检查，约 1 分钟（第二十四轮加了 12 项：浅水极限的动量和动能恒等式、ŝ 的回归值、求积收敛、入渗分布族的两端；第二十三轮加了 8 项：宽管一阶系数、动量平衡恒等式、相似解的斜率、闭合系数；第二十一轮加了 13 项：任意形状断面的层流、含渗流的阈值断面、给定管内水头的渗流计算；第二十二轮加了 16 项：σ→1 的极限、含渗流的局部律和耦合解、动量系数和惯性项、不同初值） |
| `data/` | 从 Pol 2022 摘录的试验表、砂性参数和数字化断面；Van Beek 等 2015 表 5 的平衡状态 |

运行：

```
cd code
python verify_paper1.py
python run_section.py --out ../tmp
python run_bifurcation.py --out ../tmp
python run_pol_box.py --out ../tmp
python compare_pol_heads_all.py ../Data/Pol_4TU --out ../tmp                       # 自成形断面，三种汇
python compare_pol_heads_all.py ../Data/Pol_4TU --out ../tmp --w0 30 --sinks wide  # 管宽取 30 d50
python run_okamura.py --out ../tmp
python pipe_width_data.py ../Data/Pol_4TU --out ../tmp
python run_scales.py --out ../tmp --pol-states ../tmp/paper1_pol_states.csv   # 先跑过 compare_pol_heads_all.py
python progression_rate_data.py ../Data/Pol_4TU --out ../tmp
python run_tip_criterion.py --pol-states ../tmp/paper1_pol_states.csv --out ../tmp
python run_tip_same_sand.py --out ../tmp                       # 加 --small-footprint 0.040 看占宽敏感性
python run_large_scale.py --pol-states ../tmp/paper1_pol_states.csv --out ../tmp
python run_tip_nonlocal.py --pol-states ../tmp/paper1_pol_states.csv --out ../tmp
python tip_profile_data.py ../Data/Pol_4TU --out ../tmp
python run_large_scale_width.py --out ../tmp                   # 加 --taper 算自洽宽度
python run_large_scale_width.py --cap 5.28 --tag _capx --out ../tmp   # 深度上限取管尖宽度 / 2β（5.28 倍粒径）
python run_scales.py --cap 5.28 --out ../tmp                         # 中尺度，同一条规则
python large_scale_data.py ../Data/data --out ../tmp
python make_figure_regimes.py                                   # 先跑 pipe_width_data.py
python make_figures_extra.py                                    # 先跑 run_pol_box.py、pipe_width_data.py
python make_figure_plan.py && python make_figure_section_data.py
python make_figure_c3.py
python make_figure_large_scale.py                               # 先跑 large_scale_data.py 和 run_large_scale_width.py（含 --taper）          # 读原始 xlsx 约 1 分钟
python laser_sections.py ../Data/Pol_4TU --out ../tmp          # 原始激光断面（需要 pressure/*_laser.MAT）
python laser_vs_theory.py ../Data/Pol_4TU --out ../tmp         # 逐断面对比（需要 pressure/*_pressure.MAT 和观测表）
python make_figure_laser.py ../Data/Pol_4TU
python run_section_seepage.py --out ../tmp
python run_seepage_coupled.py --out ../tmp                     # 需要 tmp/paper1_laser_vs_theory.csv 和 paper1_pol_states.csv
python run_section_convergence.py --out ../tmp
python run_assumptions.py --out ../tmp                         # 需要 tmp/paper1_pol_states.csv
python run_inertia.py --out ../tmp                             # 同上，约 10 分钟
python make_figure_inertia.py                                   # 先跑 run_inertia.py
python make_figures_jfm.py               # 稿件现用的 7 张图；先跑过上面几条（尤其 make_figure_laser.py、run_inertia.py）；然后在 manuscript/ 下编译 jfm_draft.tex
python make_paper_figures.py              # 旧版图，稿件不再引用
```
