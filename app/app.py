# import gradio as gr
# import cv2
# import numpy as np
# import datetime
# import os
#
# # # ============ 配置参数 ============
# PIXELS_PER_MM = 12
# MIN_AREA = 100
# ADAPTIVE_BLOCK_SIZE = 11
# ADAPTIVE_C = 2
# MORPH_KERNEL_SIZE = 3
# MORPH_ITERATIONS = 5
#
#
# # # ============ 裂缝检测函数 ============
# def detect_crack(image):
#     if image is None:
#         return None,"wrong , can not read the picture"
#
# #gray->blurred->adaptive->kernel->close
#
#     gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
#
#     blurred = cv2.GaussianBlur(gray , (5 , 5) , 0)
#
#     edges = cv2.Canny(blurred , 80 , 200)
#
#
#     adaptive = cv2.adaptiveThreshold(
#         edges , 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         ADAPTIVE_BLOCK_SIZE,
#         ADAPTIVE_C
#     )
#
# #=========形态学操作这里还是有点不太熟======
#     kernel = np.ones((MORPH_KERNEL_SIZE , MORPH_KERNEL_SIZE ), np.uint8)
#     closed = cv2.morphologyEx(adaptive , cv2.MORPH_CLOSE, kernel , iterations = MORPH_ITERATIONS)
#
#     contours , _ = cv2.findContours(closed , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
#
# ##========计算+画图========
#     img_copy = image.copy()
#     crack_count = 0
#     report_lines = []
#
#     for i , contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#
#         if area > MIN_AREA:
#             length = cv2.arcLength(contour , True)
#             length_mm = length / PIXELS_PER_MM
#             width = area / (length / 2)
#             width_mm = width / PIXELS_PER_MM
#
#             report_lines.append(f"裂缝{crack_count}: 长度={length_mm:.2f}mm, 宽度={width_mm:.2f}mm")
#             #画轮廓
#             cv2.drawContours(img_copy , [contour] , -1 ,(0 , 255 , 0) , 2)
#
#             crack_count += 1
#
#     if crack_count == 0:
#         report = "未检测到有效裂缝（面积 < 100 像素²）"
#     else:
#         report = f"共检测到 {crack_count} 条裂缝"
#
#
#         report += " ".join(report_lines)
#
#         report += f"\n检测时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#         report += f"标定参数：{PIXELS_PER_MM}像素 / mm（需根据实际标定调整）"
#         return img_copy,report
#
# # ============ 创建界面 ============
# with gr.Blocks(title = "AI裂缝检测系统") as demo:
#     gr.Markdown("AI裂缝检测系统")
#     gr.Markdown("上传裂缝照片，自动检测裂并计算长度、宽度")
#
#     with gr.Row():
#         with gr.Column():
#             input_img = gr.Image(label = "上传裂缝图片" , type = "numpy")
#             detect_btn = gr.Button("开始检测" , variant = "primary")
#
#         with gr.Column():
#             output_img = gr.Image(label = "检测结果")
#             output_text = gr.Textbox(label = "检测报告" , lines = 10)
#
#     detect_btn.click(
#         fn = detect_crack,
#         inputs = input_img,
#         outputs = [output_img , output_text]
#     )
#
#     gr.Markdown("====")
#     gr.Markdown(
#         "作者 : 棒 棒 | 学校 : 索菲亚大学 | Github : [crack-detector](https://github.com/12345678875/crack-detector)"
#     )
#
# # ============ 启动应用 ============
# if __name__ == "__main__":
#     demo.launch()

import gradio as gr
import torch
from PIL import Image
import sys
import os

from huggingface_hub.cli.spaces import dev_mode
from torchvision import transforms

#=======配置路径=======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR , 'results' , 'crack_model.pth')

print(f"项目根目录：{BASE_DIR}")
print(f"模型路径：{MODEL_PATH}")

#======加载模型====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在加载模型...(设备：{device}")

try:
    sys.path.append(BASE_DIR)
    from models.crack_cnn import load_model
    model = load_model(MODEL_PATH , num_classes = 2 , device = device)
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败：{e}")
    print("提示 ： 请确保results/crack_model.pth 存在")
    model = None


#=====图像预处理===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#====预测函数===
def predict_crack(image):
    if model is None:
        return "模型未加载 ， 请先训练模型"
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs , dim = 1)
            confidence , predicted = torch.max(probabilities , 1)
            labels = {0: "🔴 有裂缝", 1: "🟢 无裂缝"}
            result = labels[predicted.item()]
            confidence_score = confidence.item() * 100
            return result , f"{confidence_score:.2f}%"
    except Exception as e:
        return "预测失败" , f"错误：{str(e)}"

#=====Gradio====
with gr.Blocks(title = "AI裂缝检测系统") as demo:
    gr.Markdown("AI裂缝检测系统")
    gr.Markdown("基于深度学习的混泥土裂缝自动检测")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label = "上传图片" , type = "pil" , height = 300)
            submit_btn = gr.Button("开始检测" , variant = "primary")
        with gr.Column():
            result_label = gr.Textbox(label = "检测结果")
            confidence_score = gr.Textbox(label = "置信度")

    submit_btn.click(
        fn = predict_crack,
        inputs = input_image,
        outputs = [result_label , confidence_score]
    )

if __name__ == "__main__":
    demo.launch()

