# 指针式喷灌机变量灌溉处方图生成工具：Streamlit 大图上传优化版 v3

## 这版解决什么问题

这版专门针对 Streamlit 部署时上传无人机 GeoTIFF 慢、读取慢、内存占用大的问题做了改造：

1. 上传后先保存为服务器临时文件，不再用 MemoryFile 一次性在内存里读取整张图。
2. 使用 rasterio window 分块读取，每次只读一个窗口。
3. 第一遍抽样估算 NDVI、NDRE、GNDVI、EVI/EVI2 和热红外温度分位数。
4. 第二遍分块计算目标灌水量，并直接聚合到“角度扇区 × 喷头编号”矩阵。
5. 不再输出完整像元点表，避免生成几百万行甚至上亿行 CSV。
6. 使用 st.form，只有点击按钮后才计算，避免调整参数时反复重算。
7. 自动带 `.streamlit/config.toml`，部署到 Streamlit Cloud 后可把单文件上传上限提高到 2048 MB。

## 重要说明

这版可以显著降低后端读取和计算压力，但不能突破浏览器和网络本身的上传速度限制。对于几百 MB 的裁剪影像可以使用；如果是数 GB 级无人机正射影像，正式软件仍建议采用阿里云 OSS 分片上传 + ECS 后台分块计算。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 部署

GitHub 仓库结构应为：

```text
你的仓库/
├─ app.py
├─ requirements.txt
├─ .streamlit/
│  └─ config.toml
└─ README_部署说明.md
```

部署时 Main file path 填：

```text
app.py
```

## 影像输入要求

必选：Red、Green、RedEdge、NIR 单波段 GeoTIFF。

可选：Blue 单波段 GeoTIFF，上传后计算 EVI；不上传则计算 EVI2。

可选：Thermal 热红外 GeoTIFF，上传后计算 CWSI。

所有波段必须已经完成配准，行列数一致，空间范围一致。正式处方图建议先裁剪到指针机覆盖圆形区域，并降采样到 0.3 到 1.0 m。

## 输出文件

生成 ZIP 中包含：

- `01_sprinkler_parameters.csv`：喷头参数表
- `02_polar_target_depth.csv`：角度扇区 × 喷头目标灌水量
- `03_sector_speed_prescription.csv`：角度扇区行走速度处方
- `04_sprinkler_control_long.csv`：控制器用喷头电磁阀长表
- `05_duty_matrix_wide.csv`：人工检查用占空比矩阵
- `06_index_percentile_stats.csv`：指数分位数统计
- `07_run_summary.csv`：运行参数摘要

