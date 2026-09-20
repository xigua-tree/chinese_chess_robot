# 标定数据

此目录存放摄像头和机械臂的标定数据。每个硬件环境需要重新标定，请参考 docs/calibration.md。

## 文件说明

| 文件 | 说明 |
|------|------|
| map_model.npz | TPS 薄板样条映射模型 |
| pixel_to_world.npy | 像素坐标→世界坐标映射 |
| calibration_matrix.npy | 标定矩阵 |
| corner_points.npy | 角点坐标 |
