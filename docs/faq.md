# 常见问题

## 摄像头

**Q: 摄像头打不开？**
A: 检查 `VideoStreamThread` 中的 `camera_idx` 参数。Windows 上通常为 0 或 1。可以用 `tools/camera_settings.py` 测试。

**Q: 画面太暗/太亮？**
A: 调节 `VideoStreamThread` 中的曝光参数：
```python
self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 关闭自动曝光
self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)        # 手动曝光值
```

## 视觉识别

**Q: 棋子识别不准确？**
A: 按优先级排查：
1. 用 "Hough Controls" 调参确保所有棋子被检测到
2. 检查 `CONF_THRESH` 置信度阈值（默认 0.70）
3. 检查棋盘四角标定是否准确
4. 检查光照是否均匀

**Q: 走法检测经常失败？**
A: 增大 `VOTE_FRAMES` 和 `VOTE_THRESHOLD` 提高帧间投票精度。

## 机械臂

**Q: 串口通信失败？**
A: 检查 `robot_arm/arm_controller.py` 中的 `COM9` 是否匹配你的设备。用设备管理器查看串口号。

**Q: 机械臂不到位？**
A: 检查 Z 轴脉冲参数 `Z_UP`/`Z_DOWN` 是否合适。可能需要重新标定 TPS 映射模型。

## 引擎

**Q: Pikafish 引擎启动失败？**
A: 确认 `engines/pikafish/` 下存在 `pikafish-avx2.exe` 和 `pikafish.nnue`。

**Q: 引擎走法不合法？**
A: 系统会自动调用摄像头重新识别局面并重建引擎状态。如果持续失败，可能是识别结果错误导致 cchess 状态不一致。

## 语音

**Q: 语音没有声音？**
A: 检查 `voice/vits_model/` 下是否有模型文件，以及声卡是否正常工作。
