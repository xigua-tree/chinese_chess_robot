# 系统架构

## 概述

象棋机器人由 5 个核心模块组成，通过 TCP 协议在 localhost:8888 上通信。

## 模块关系图

```
┌──────────────────────────────────────────────────────┐
│                    Qt 主控界面 (gui/)                  │
│               TCP Server (localhost:8888)             │
└────┬──────────┬──────────────┬────────────┬──────────┘
     │          │              │            │
     ▼          ▼              ▼            ▼
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐
│象棋引擎  │ │机械臂控制 │ │ 语音播报  │ │ 五子棋   │
│chess_   │ │robot_arm│ │  voice   │ │ gomoku   │
│engine   │ │         │ │          │ │          │
└────┬────┘ └─────────┘ └──────────┘ └──────────┘
     │
     ▼
┌─────────┐
│ 视觉模块 │
│ vision  │
└─────────┘
```

## 各模块详解

### 象棋引擎模块 (chess_engine/)

**职责**：对弈逻辑核心，协调视觉识别和机械臂执行。

**核心类**：
- `Config`：全局配置（棋盘坐标、模型路径、棋子映射表、FEN 校验）
- `ProtocolClient`：TCP 通信，阻塞等待消息，支持前缀匹配
- `EngineController`：Pikafish UCI 引擎进程管理，含崩溃自动恢复
- `GameController`：对弈状态机，支持开局/残局/悔棋/替思/摆盘
- `MoveRecord`：单步走法记录

**状态机流程**：
```
INIT → READY → WAIT_START → HUMAN_TURN ⇄ ENGINE_TURN → GAME_OVER
                              ↑              ↑
                              └── UNDO ──────┘
                              ↑
                              └── AUTO_SETUP
```

### 视觉模块 (vision/)

**职责**：摄像头采集、棋盘识别、棋子定位。

**核心类**：
- `VideoStreamThread`：独立线程采集摄像头，实时透视变换和显示
- `VisionController`：
  - `recognize_board()`：单帧 HoughCircles 寻圆 + YOLO 分类
  - `capture_stable_board()`：多帧投票 + 棋子数量校验
  - `get_move()`：前后局面差分 → UCI 走法
  - `recognize_waste_area()`：废棋区多帧识别

**识别流程**：
1. 摄像头捕获 1280×720 原始帧
2. 透视变换 → 707×630 俯视图
3. 霍夫圆检测定位棋子圆心
4. 每个圆心区域 48×48 → YOLO 分类 → 14 类棋子标签
5. 圆心坐标映射到 10×9 网格
6. 多帧投票（15帧，阈值8票）确认稳定局面

### 机械臂模块 (robot_arm/)

**职责**：三轴步进电机控制，棋子物理移动。

**核心功能**：
- 逆运动学：XY 世界坐标 → 关节角 → 电机脉冲
- TPS 映射：图像像素坐标 → 世界坐标
- 电磁铁控制：DTR 信号控制抓取/释放
- 运动补偿：根据目标距离自动调整 Z 轴高度

### 语音模块 (voice/)

**职责**：TCP 监听文字消息，VITS TTS 合成语音播报。

### 五子棋模块 (gomoku/)

**职责**：五子棋引擎 + 视觉识别的独立子系统。

## 数据流

```
摄像头帧 → warpPerspective(四角标定矩阵) → 707×630 俯视图
  → HoughCircles → 棋子圆心 (cx, cy, r)
  → YOLO 分类 → 棋子标签 (R_Kin, B_Paw, ...)
  → 坐标映射 → 10×9 棋盘矩阵
  → 差分前后帧 → removed/added/changed 列表
  → UCI 走法字符串 (如 "h2e2")
  → cchess 规则校验 → is_legal()
  → Pikafish 引擎 → bestmove
  → uci_to_pixel() → 透视图像素坐标
  → TPS 映射 → 世界坐标 (wx, wy)
  → 逆运动学 → 关节角 (a1, a2)
  → 电机脉冲 → 串口发送
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 视觉 | OpenCV, YOLO (ultralytics), HoughCircles |
| 引擎 | Pikafish (UCI 协议), cchess (规则校验) |
| 控制 | pyserial, 自定义逆运动学 |
| 语音 | sherpa-onnx (VITS), edge-tts |
| 通信 | TCP socket (纯文本协议) |
| 界面 | Qt (C++ / PyQt) |
