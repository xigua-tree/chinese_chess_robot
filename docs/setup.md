# 环境搭建指南

## 系统要求

- Windows 10/11 64 位
- Python 3.13+
- USB 摄像头
- RS485 串口（用于机械臂通信）
- 至少 8GB 内存

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/chess_robot.git
cd chess_robot
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装中国象棋规则库

```bash
cd python-chinese-chess
pip install -e .
cd ..
```

### 5. 下载模型和引擎

#### 棋子分类模型

从 [Releases]() 下载 `chess_classifier.pt`，放入 `models/` 目录。

#### Pikafish 象棋引擎

从 [Pikafish Releases](https://github.com/Pikafish/Pikafish/releases) 下载 Windows 版本：
- `pikafish-avx2.exe` → `engines/pikafish/`
- `pikafish.nnue` → `engines/pikafish/`

#### VITS 语音模型

从 ModelScope 下载 `vits-aishell3.onnx`：
- 下载地址：https://www.modelscope.cn/models/iic/speech_tts_vits_zh_aishell3
- 将模型文件放入 `voice/vits_model/`

#### Rapfi 五子棋引擎

从 [Rapfi Releases](https://github.com/Rapfi/Rapfi/releases) 下载 Windows 版本：
- `pbrain-rapfi_avxvnni.exe` → `engines/pbrain-rapfi/`

### 6. 配置硬件

#### 摄像头

默认使用摄像头索引 0。可在 `chess_engine/config.py` 中修改：

```python
video_stream = VideoStreamThread(camera_idx=0)  # 改为你的摄像头索引
```

#### 串口

在 `robot_arm/arm_controller.py` 中修改串口号：

```python
ser = serial.Serial("COM9", 115200, timeout=0.05)  # 改为你的串口号
```

### 7. 标定

首次使用需要标定，参考 [calibration.md](calibration.md)。

## 验证安装

```bash
# 测试摄像头
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.read()[0])"

# 测试象棋规则库
python -c "import cchess; print(cchess.Board().fen())"

# 测试串口
python -c "import serial; print(serial.tools.list_ports.comports())"
```
