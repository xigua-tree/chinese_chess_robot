# from ultralytics import YOLO
# import os

# def main():
#     # 1. 加载预训练的 YOLOv8n 分类模型
#     # 如果你是第一次运行，它会自动下载 yolov8n-cls.pt
#     model = YOLO('yolov8n-cls.pt')

#     # 2. 开始训练
#     # data: 指向包含 train 和 val 文件夹的根目录
#     # epochs: 训练轮数，100轮对小数据集比较合适
#     # imgsz: 输入图片大小，你的切片是40x40，这里设为64或128可以让模型学习更细致，或者设为224（默认）
#     results = model.train(
#         data=r'E:\chess_robot\chess_classify_data',
#         epochs=100,
#         imgsz=64,       # 因为你的原始切片较小，设为64即可
#         batch=32,       # 每次处理32张图
#         name='chess_r_test2', # 实验名称
#         device='0'    # 如果你之前检查 cuda.is_available() 是 False，这里用 'cpu'
#                         # 如果你有 NVIDIA 显卡并装好了环境，可以改为 device=0
#     )

#     print("训练完成！模型保存在: runs/classify/chess_r_test2/weights/best.pt")

# if __name__ == '__main__':
#     main()

from ultralytics import YOLO

# 1. 加载你训练好的 pt 模型
model = YOLO('best.pt')

# 2. 导出为 onnx 格式（建议指定 imgsz，针对象棋盘建议 640x640）
model.export(format='onnx', imgsz=640, opset=12)