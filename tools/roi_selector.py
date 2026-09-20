import cv2
import numpy as np
import os

def get_roi_from_image(file_path):
    # 1. 稳健读取图片
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # 2. 弹出窗口让你框选 ROI
    # selectROI 参数说明:
    # "Select ROI": 窗口名称
    # img: 要选择的图像
    # showCrosshair=True: 显示十字准星
    # fromCenter=False: 不从中心开始框选
    print(">>> 请在弹出的窗口中，用鼠标框选【整个棋盘区域】，选好后按回车(Enter)或空格")
    roi_box = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
    
    # roi_box 的格式是 (x, y, w, h)
    x, y, w, h = roi_box

    if w > 0 and h > 0:
        # 3. 截取图像 (矩阵切片: img[y1:y2, x1:x2])
        roi_img = img[y:y+h, x:x+w]
        
        # 4. 显示并保存截取后的区域
        cv2.imshow("Cropped ROI", roi_img)
        cv2.imwrite("board_roi.jpg", roi_img) # 保存下来供后续程序使用
        print(f"ROI 已截取并保存为 board_roi.jpg, 坐标: x={x}, y={y}, 宽={w}, 高={h}")
        
        print("\n下一步提示: 以后我们所有的识别程序都只处理 board_roi.jpg，背景就干扰不到你了。")
        cv2.waitKey(0)
    else:
        print("未选择有效的区域。")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 指向你的第五张图片
    file_path = os.path.join("e2", "5.png")
    get_roi_from_image(file_path)