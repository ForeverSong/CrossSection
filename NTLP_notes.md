# NTLP 代码学习笔记（面向 BEP 两相流）

对象：<https://github.com/RichterLab/NTLP> `master`，提交 `e912f3a`（2026-11-19 克隆）。
行号均指该提交。NTLP = NCAR Turbulence with Lagrangian Particles，Fortran + MPI，约 2.5 万行。

## 1. 它是什么，不是什么

- 是：三维不可压 Navier–Stokes（DNS 或 LES）+ 双向耦合的拉格朗日**点粒子**；水平方向周期、伪谱；
  垂向二阶有限差分；为海雾、海洋飞沫、云室（Pi chamber）而写，粒子是空气中的液滴。
- 不是：稠密颗粒流求解器。没有颗粒–颗粒接触（只有可选的液滴并合 `particle_coalesce`），
  没有体积分数（流体方程里不含孔隙率），没有浮力、附加质量、压力梯度力（气–液密度比 ~1000，全部可忽略）。
- 网格规模是**编译期**常量：`defs.F:24` 的 `maxnx, maxny, maxnz`，改分辨率要重新编译。
- README 写明需要 Intel 编译器（Makefile 用 `mpiifort -fc=ifx`）和 NetCDF；教程针对 Notre Dame 集群。

## 2. 文件地图

| 文件 | 行数 | 作用 |
|---|---|---|
| `les.F` | 10 892 | 主程序、时间推进、流体右端项、压力求解、MPI 转置、统计、I/O |
| `particles.f90` | 5 484 | 粒子的全部内容：插值、推进、边界、双向耦合、MPI 迁移、液滴热力学 |
| `defs.F` | 473 | 全局模块 `pars / fields / con_data / con_stats / fftwk` |
| `fft.f` | 3 011 | FFTPACK |
| `kdtree.f90` | 556 | 近邻搜索（并合用） |
| `data_structures.f90`, `measurement.f90` | 684 / 1 350 | 计时与性能剖析 |
| `netcdf_io.f90`, `tec_io.f90` | 1 545 / 565 | 输出 |
| `test_cases/channel/params.in` | — | **最该先看的算例**：槽道 DNS + 10 万粒子 |

## 3. 主循环（`les.F` 第 1 行起的 `program les_mpi_2d`）

每个时间步 3 个 Runge–Kutta 子步，系数在 `setcon`（`les.F:856-861`）：

```
zetas = (0, -17/60, -5/12)      gama = (8/15, 5/12, 3/4)
```

```
do 每个时间步
  do istage = 1, 3
     comp1(istage)   les.F:1369  右端项 -> rhs_uvw_DNS (1814) / rhs_uvw (1531, LES)
                                  粒子反馈 partsrc 在这里加进动量方程 (1900-1905)
     comp_p          les.F:3659  组装压力泊松方程右端 -> pressure (3911)
                                  x、y 方向 FFT (fft2d_mpi 4401) -> xtoz_trans (10060)
                                  -> 每个波数一个三对角方程 solve_trid (4008) -> ztox_trans (10132)
     comp2           les.F:3789  用压力梯度修正速度；dealias (8193)
     particle_update_rk3(istage)  仅当 ipart_method = 1
  end do
  particle_update_BE              ipart_method = 2（channel 算例用这个）
  particle_coupling_update        particles.f90:1151  粒子 -> 网格动量源
  particle_coupling_exchange      particles.f90:1066  跨进程汇总源项
  apply_particle_coupling         particles.f90:1278  温度、湿度源直接加到场上
  particle_coalesce / SFS_velocity / particle_reintro   可选
  particle_xy_stats               统计
end do
```

## 4. 流体部分要点

- 变量布置：`u, v, p` 在半层 `zz(k)`，`w` 在整层 `z(k)`；水平方向同位、谱导数（`xderivp`、`yderiv`）。
- 垂向网格可拉伸：`vgrid_channel`（`les.F:5682`，`iz_space=1`）在上下壁面加密。
- DNS 的壁面条件在 `lower_dns / upper_dns`（3172 / 3239）；Couette 流用 `Uo`，压力驱动槽道流用 `dpdx`。
- MPI：每个进程持有完整的 x 范围、y–z 的一块；FFT 和三对角求解之间靠 `xtoy/ytox/xtoz/ztox_trans` 转置。
  这一大块（`les.F:9764-10438`）在单 GPU 上**完全不需要**。

## 5. 粒子部分要点

- 数据结构：`type particle`（`particles.f90:76`）+ 双向链表，逐粒子指针遍历。
  字段含位置 `xp`、速度 `vp`、插值得到的流体速度 `uf`、上一步速度 `vp_old`、半径、温度、`mult`（一个计算粒子代表的真实粒子数）。
- 插值：`uf_interp_lin`（767，三线性）或 `uf_interp`（554，六阶拉格朗日）；
  之前要 `fill_ext`（91）把流场扩一圈幽灵层。
- 推进：`particle_update_BE`（3000）是后向欧拉：

  ```
  xp = xp + dt*vp
  vp = (vp + dt*uf/tau_p + dt*g) / (1 + dt/tau_p),   tau_p = rho_p d^2 / (18 rho_a nu)
  ```

  `particle_update_rk3`（2845）与流体同一套 RK3，带 Schiller–Naumann 修正 `1 + 0.15 Re_p^0.687`。
- 边界：`particle_bcs_nonperiodic`（2683）上下壁面弹性反射，`icase≠0` 时触底即销毁；
  `particle_bcs_periodic`（2814）水平周期。
- 双向耦合：`particle_coupling_update`（1151）把 `-m_p/rho_a * (dv_p/dt - g) * mult / dV`
  按三线性权重分给周围 8 个节点，其中 `dv_p/dt = (vp - vp_old)/dt`。
- MPI：粒子用另一套按粒子数均衡的区域分解，`particle_exchange`（1435）做跨进程迁移。

## 6. NTLP ↔ `bepflow/pipe_el2d.py` 对照

| NTLP | 本项目（PyTorch，二维 x–z） | 改动 |
|---|---|---|
| `comp1` / `rhs_uvw_DNS` | `Flow.rhs` | 去掉 y 方向、标量、LES |
| `comp_p` / `pressure` / `solve_trid` | `Flow.project` | 每个波数的 nz×nz 系统预先求逆，一次批量矩阵乘 |
| `comp2` | `Flow.project` 末尾 | — |
| RK3 系数 `zetas, gama` | `RK_ZETA, RK_GAMMA` | 相同 |
| `fill_ext` | `Flow.u_ext` | 幽灵层取 −u，壁面处精确为 0 |
| `uf_interp_lin` | `Particles.interp` | 链表逐粒子 → 整个数组一次 gather |
| `particle_update_BE` | `Particles.update` | 加浮力折减重力、附加质量、Schiller–Naumann |
| `particle_bcs_nonperiodic` | `Particles.update` 末尾 | 加“沉积到床面”选项 |
| `particle_coupling_update` | `Particles.project` | `index_add_`；阻力在施加边界条件之前计算 |
| `part%mult` | `Particles.mult` | 相同概念 |
| `particle_exchange`、所有 `*_trans` | 无 | 单 GPU 不需要 |

## 7. 对 BEP 的取舍

**直接沿用的思想**：欧拉–拉格朗日总体结构、网格↔粒子的插值与投影、后向欧拉处理刚性阻力、`mult` 加权、RK3 分步投影。

**必须改造**：

1. 水–砂密度比 2.65：浮力、附加质量、压力梯度力都要进粒子方程；NTLP 的 `part_grav` 是不含浮力的全重力。
2. 床面：NTLP 只有反射或销毁；BEP 需要沉积、再起动和床面演化。
3. 粒径：管道深度只有几到几十个粒径，点粒子假设（粒径小于网格）经常不成立，
   要换成体积平均（非解析 CFD-DEM）或颗粒解析方法。
4. 稠密接触：推移质输运由颗粒间接触主导，需要 DEM 接触模型和 GPU 邻居搜索。

**可以舍弃**：液滴热力学与蒸发（`particles.f90` 约 4000 行以后的大部分）、湿度/温度标量、
LES 亚格子模型、辐射、Coriolis、Monin–Obukhov 壁面模型、全部 MPI 代码。

## 8. 建议的阅读顺序（约两周）

1. `documentation/explanation/ntlp-overview.md`（10 分钟）。
2. `test_cases/channel/params.in`：每个开关对应代码里的一个分支，边读边在 `les.F` 里搜。
3. 主循环 → `comp1` → `rhs_uvw_DNS` → `comp_p` → `pressure` → `comp2`。
4. `particles.f90`：`type particle` → `uf_interp_lin` → `particle_update_BE` → `particle_bcs_nonperiodic` → `particle_coupling_update`。
   每读完一个就对照 `pipe_el2d.py` 里的同名方法。
5. 想在本机跑：Windows 上用 WSL2 + Intel oneAPI（免费）+ NetCDF-Fortran，先把 `defs.F` 的
   `maxnx/maxny/maxnz` 调小跑 `channel` 算例。gfortran 不在官方支持范围内，能否编过我没有验证。

## 9. 读代码时容易踩的点

- `particle_update_BE` 里先算了 `corrfac = 1 + 0.15 Re_p^0.687`，紧接着又写了 `corrfac = 1.0`——
  后向欧拉分支实际用的是纯 Stokes 阻力。
- `particle_update_rk3` 开头的注释写着“很久没更新”，以 BE 分支为准。
- 双向耦合的 `vrhs` 用的是边界处理之后的速度差，壁面反射造成的速度跳变也会被算进反馈。
- `partsrc` 的第 3 分量存放在 u、v 的位置上，加进 w 方程前要垂向插值（`les.F:1904`）。
