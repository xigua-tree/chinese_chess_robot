# 象棋机器人开源重构设计文档

## 概述

将 2026 英特尔杯"实里桌通用棋类机器人"项目重构为适合开源学习的工程结构。

**目标定位**：技术参考/学习资料，代码清晰 + 文档详尽，全部中文。

**原则**：
- 只保留每个模块的最新版本，删除所有旧版本和备份
- 目录名英文规范化，README 和文档全部中文
- 保持所有子系统功能完整

---

## 新目录结构

```
chess_robot/
├── README.md                         # 项目介绍、快速开始、硬件清单
├── LICENSE                           # 开源协议（MIT）
├── requirements.txt                  # Python 依赖
├── .gitignore                        # 完善的忽略规则
├── Run.bat                           # 一键启动脚本（相对路径）
│
├── docs/                             # 文档
│   ├── architecture.md               # 系统架构说明
│   ├── setup.md                      # 环境搭建指南
│   ├── calibration.md                # 摄像头标定 & 机械臂校准
│   ├── protocol.md                   # 模块间 TCP 通信协议
│   └── faq.md                        # 常见问题
│
├── vision/                           # 视觉识别模块
│   ├── chess_vision.py               # 象棋棋盘识别（VisionController）
│   ├── video_stream.py               # 摄像头视频流线程
│   └── board_utils.py                # 透视变换、网格绘制、坐标转换
│
├── chess_engine/                     # 象棋引擎模块
│   ├── engine_controller.py          # UCI 引擎进程管理
│   ├── game_controller.py            # 对弈状态机
│   ├── protocol_client.py            # TCP 通信客户端
│   └── config.py                     # 全局可调参数
│
├── robot_arm/                        # 机械臂控制
│   ├── arm_controller.py             # 机械臂控制核心
│   ├── kinematics.py                 # 逆运动学 + 电机角度转换
│   ├── tps_mapper.py                 # 像素→世界坐标映射
│   ├── relay_control.py              # 继电器/电磁铁控制
│   ├── tests/                        # 测试脚本
│   │   ├── test_kinematics.py        # 角度/逆运动学测试
│   │   ├── test_click_reach.py       # 点击到达测试
│   │   ├── test_matrix_point.py      # 矩阵点转XY测试
│   │   └── test_rectangle.py         # 矩形验证测试
│   └── calibration/                  # 标定工具
│       ├── xy_calibration.py         # 电机XY打点
│       └── camera_matrix.py          # 获取像素坐标映射矩阵
│
├── voice/                            # 语音模块
│   ├── voice_broadcast.py            # 语音播报客户端（VITS TTS）
│   ├── voice_assets.py               # 语音素材生成（Edge TTS）
│   ├── voice_test.py                 # 音色测试工具
│   └── vits_model/                   # VITS 模型文件
│
├── gomoku/                           # 五子棋子系统
│   ├── gomoku_engine.py              # 五子棋引擎封装
│   ├── gomoku_vision.py              # 五子棋视觉识别
│   ├── gomoku_engine_binary/         # 五子棋引擎可执行文件
│   └── utils/                        # 辅助工具
│       ├── perspective_transform.py  # 矩形变换
│       └── hough_debug.py            # 霍夫圆参数调试
│
├── gui/                              # Qt 界面（源码）
│
├── model_train/                      # 模型训练
│   ├── train_classifier.py           # 棋子分类器训练脚本
│   └── README.md                     # 训练说明
│
├── models/                           # 训练好的模型
│   ├── chess_classifier.pt           # 棋子分类模型（best.pt）
│   └── chess_classifier.onnx         # ONNX 导出模型
│
├── calibration_data/                 # 标定数据
│   ├── map_model.npz                 # TPS 映射模型
│   ├── pixel_to_world.npy            # 像素转世界坐标
│   ├── calibration_matrix.npy        # 标定矩阵
│   └── corner_points.npy             # 角点坐标
│
├── engines/                          # 第三方引擎
│   ├── pikafish/                     # Pikafish 象棋引擎
│   └── pbrain-rapfi/                 # Rapfi 五子棋引擎
│
└── tools/                            # 调试工具
    ├── hough_tuner.py                # 霍夫圆参数实时调节
    ├── click_calibrate.py            # 鼠标点击标定工具
    ├── camera_settings.py            # 摄像头画面设置
    ├── roi_selector.py               # 感兴趣区域获取
    ├── tcp_example.py                # TCP 连接示例
    └── serial_test.py                # 串口通信测试
```

---

## 模块职责

### vision/ — 视觉识别
- 摄像头实时采集（USB 摄像头，1280×720）
- 透视变换（四角标定 → 707×630 俯视图）
- 霍夫圆检测定位棋子
- YOLO 分类模型识别棋子类型
- 帧间投票提高识别稳定性
- 前后局面差分计算人类走法
- 废棋区识别

### chess_engine/ — 象棋引擎
- UCI 协议封装 Pikafish 引擎
- 对弈状态机（INIT → WAIT_START → HUMAN_TURN ↔ ENGINE_TURN → GAME_OVER）
- 悔棋（撤销两步：人+引擎）
- 替思（引擎替人类思考）
- 一键摆盘（自动摆放残局/开局）
- cchess 规则引擎校验走法合法性
- TCP 协议与服务端通信

### robot_arm/ — 机械臂控制
- 三轴串口电机控制（XY 角度 + Z 脉冲）
- 逆运动学（XY 坐标 → 关节角）
- TPS 薄板样条映射（像素 → 世界坐标）
- 电磁铁抓取/释放
- 棋子移动完整流程（抓取→抬起→移动→放下→释放）

### voice/ — 语音播报
- VITS 中文语音合成（sherpa-onnx）
- TCP 监听服务端文字消息
- Edge TTS 语音素材预生成
- 多音色对比测试

### gomoku/ — 五子棋
- Rapfi 五子棋引擎封装（Pbrain 协议）
- 棋盘视觉识别（霍夫圆 + 透视变换）
- 五子棋规则判断（五连检测）

---

## 数据流

```
摄像头 → VideoStreamThread → 透视变换(warped)
warped → VisionController.recognize_board() → 10×9 棋盘矩阵
10×9矩阵 → VisionController.get_move() → UCI 走法
UCI走法 → cchess 规则校验 → 同步到 Pikafish 引擎
Pikafish → EngineController.get_best_move() → 引擎 UCI 走法
UCI走法 → VisionController.uci_to_pixel() → 像素坐标
像素坐标 → TCP → robot_arm → TPS坐标映射 → 逆运动学 → 电机脉冲指令
```

## TCP 通信协议

各模块通过 localhost:8888 与 Qt 服务端通信：

| 消息 | 格式 | 说明 |
|------|------|------|
| 注册 | `REGISTER:模块名` | 模块上线注册 |
| 开始 | `开始对弈{难度}` | 服务端发起对局 |
| 残局 | `残局对弈{难度}:{名称}` | 残局模式 |
| 落子指令 | `棋子移动:(x1,y1),(x2,y2)` | 机械臂移动命令 |
| 运动完成 | `运动完成` | 机械臂反馈 |
| 请落子 | `请玩家落子` | 提示人类行动 |
| 落子完成 | `玩家落子完成` | 人类确认完成 |
| 语音 | `文字:播报内容` | 触发语音播报 |
| 悔棋 | `悔棋` | 撤销两步 |
| 退出 | `退出` | 终止程序 |

---

## 文件溯源

| 新位置 | 来源文件 |
|--------|---------|
| `vision/chess_vision.py` | `pikafish_auto/Pikafish_auto4.py` (VisionController 类) |
| `vision/video_stream.py` | `pikafish_auto/Pikafish_auto4.py` (VideoStreamThread 类) |
| `vision/board_utils.py` | `pikafish_auto/Pikafish_auto4.py` (draw_board_grid 等) |
| `chess_engine/engine_controller.py` | `pikafish_auto/Pikafish_auto4.py` (EngineController 类) |
| `chess_engine/game_controller.py` | `pikafish_auto/Pikafish_auto4.py` (GameController 类) |
| `chess_engine/protocol_client.py` | `pikafish_auto/Pikafish_auto4.py` (ProtocolClient 类) |
| `chess_engine/config.py` | `pikafish_auto/Pikafish_auto4.py` (Config 类) |
| `robot_arm/arm_controller.py` | `机械臂控制程序/机械臂控制.py` |
| `voice/voice_broadcast.py` | `pikafish_auto/语音播报.py` |
| `voice/voice_assets.py` | `pikafish_auto/语音测试.py` |
| `voice/voice_test.py` | `pikafish_auto/语音转文字.py` |
| `gomoku/gomoku_engine.py` | `五子棋引擎/五子棋引擎.py` |
| `gomoku/gomoku_vision.py` | `五子棋引擎/第二代五子棋视觉.py` |
| `model_train/train_classifier.py` | `model_train/train_chess.py` |

---

## 需要删除的文件

- 旧版本：`Pikafish_auto.py`, `Pikafish_auto2.py`, `Pikafish_auto3.py`, `照片识别步骤.py`
- 重复备份：`五子棋引擎/备份/` (7个文件), `机械臂控制程序/备份/` (4个文件), `获取矩阵/` 下 copy 文件
- 重复目录：`pikafish_auto/model_train/`, `pikafish_auto/pikafish/`, `runs/`
- 重复训练结果：`model_train/runs/classify/chess_r_test/` 和 `chess_r_test1/`
- 大型二进制：`X42_E_TTL_Tool_V1.2.4.exe`, `yolo26n.pt`, `yolov8n-cls.pt`, `QT/` (已编译)
- 无关文件：`聊天室`, `示例代码/`, `乱/`
- 调试残留：`主程序调试.py`, `test.py` (cchess 测试)

---

## 开源准备清单

- [ ] 完善 `.gitignore`（忽略 __pycache__, *.pyc, .vscode/, *.onnx 大模型等）
- [ ] 添加 `LICENSE` 文件（MIT）
- [ ] 编写 `README.md`（项目介绍、硬件清单、快速开始、模块说明）
- [ ] 编写 `requirements.txt`
- [ ] 编写 `docs/architecture.md`
- [ ] 编写 `docs/setup.md`
- [ ] 编写 `docs/protocol.md`
- [ ] 编写 `docs/calibration.md`
- [ ] 编写 `docs/faq.md`
- [ ] 更新 `Run.bat` 为相对路径
- [ ] 模型文件处理：大文件用 Git LFS 或提供下载链接
- [ ] 代码注释中文化
