# def detect_crack(image):
#     if image is None:
#         return None,"wrong , can not read the picture"
#
#     # gray->blurred->adaptive->kernel->close
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred , 80 , 200)
#     adaptive = cv2.adaptiveThreshold(
#         edges , 255 ,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,
#         cv2.THRESH_BINARY_INV,
#         ADAPTIVE_BLOCK_SIZE,
#         ADAPTIVE_C
#     )
#     kernel = np.ones((3 , 3) , np.uint8)
#     closed = cv2.morphologyEx(adaptive , cv2.MORPH_CLOSE, kernel , iterations = 3)
#     contours , _ = cv2.findContours(closed , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
# #
#
#     img_copy = image.copy()
#     crack_count = 0
#     report_lines = []
#
#     for i , contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#
#         if area > 100:
#             length = cv.arcLength(contour , True)
#             length_mm = length / 12
#             width = area / (length / 2)
#             width_mm = width / 12
#
#             report_lines.append(f"裂缝{crack_count} : 长度 = {length_mm : .2f} mm , 宽度 = {width_mm : .2f} mm")
#             cv2.drawContours(img_copy , contour , -1 , (0 , 255 , 0) , 2)
#             crack_count += 1
#
#     if crack_count == 0:
#         report = "未检测到有效裂缝"
#     else:
#         report = f"共检测到 {crack_count} 条裂缝"
#         report += " ".join(report_lines)
#         report += f"\n检测时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#         return img_copy,report

import gradio as gr
import torch
from PIL import Image
import sys
import os
import cv2
import numpy as np
# import datatime
from huggingface_hub.cli.spaces import dev_mode
from torchvision import transforms

# # ============ 配置参数 ============
PIXELS_PER_MM = 12
MIN_AREA = 100
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2
MORPH_KERNEL_SIZE = 3
MORPH_ITERATIONS = 5

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


#===轮廓图===
def detect_crack(image):
    if image is None:
        return None, "wrong , can not read the picture"

    # gray->blurred->adaptive->kernel->close
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 200)
    adaptive = cv2.adaptiveThreshold(
        edges, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #

    img_copy = image.copy()
    crack_count = 0
    report_lines = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 100:
            length = cv2.arcLength(contour, True)
            length_mm = length / 12
            width = area / (length / 2)
            width_mm = width / 12

            report_lines.append(f"裂缝{crack_count} : 长度 = {length_mm : .2f} mm , 宽度 = {width_mm : .2f} mm")
            cv2.drawContours(img_copy, contour, -1, (0, 255, 0), 2)
            crack_count += 1

    if crack_count == 0:
        report = "未检测到有效裂缝"
    else:
        report = f"共检测到 {crack_count} 条裂缝"
        report += " ".join(report_lines)
        # report += f"\n检测时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return img_copy, report

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
            labels = {1: "🔴 有裂缝", 0: "🟢 无裂缝"}
            result = labels[predicted.item()]
            confidence_score = confidence.item() * 100
            return result , f"{confidence_score:.2f}%"
    except Exception as e:
        return "预测失败" , f"错误：{str(e)}"

#=====Gradio====
with gr.Blocks(title = "AI裂缝检测系统") as demo:
    gr.Markdown(
        """
        # 🧠 AI Crack Detection System

        **基于深度学习的混凝土裂缝检测系统**

        上传图片，AI 将自动判断是否存在裂缝，并给出预测置信度。
        """
    )
    gr.Markdown("上传裂缝照片，自动检测裂并计算长度、宽度")
    status = gr.Markdown("系统已启动，等待图片上传...")


    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label = "上传裂缝图片" , type = "numpy")
            detect_btn = gr.Button("开始检测" , variant = "primary")

        with gr.Column():
            output_img = gr.Image("检测结果")
            output_text = gr.Textbox(label = "检测报告" , lines = 15)

    detect_btn.click(
        fn = detect_crack,
        inputs = input_img,
        outputs = [output_img , output_text]
    )


    with gr.Row():
        with gr.Column():
                with gr.Group():
                    gr.Markdown("### 上传检测图片")
                    input_image = gr.Image(label = "上传图片" , type = "pil" , height = 300)
                    submit_btn = gr.Button("开始检测" , variant = "primary")

    with gr.Column():
        with gr.Group():
            gr.Markdown("###检测结果")
            result_label = gr.Textbox(label = "检测结果" )
            confidence_score = gr.Textbox(label = "可信度" )

    submit_btn.click(
            fn = predict_crack,
            inputs = input_image,
            outputs = [result_label , confidence_score ]
        )
    gr.Markdown("--------------------------")
    gr.Markdown("###示例图片")

    gr.Examples(
        examples = [
            ["examples/crack1.jpg"],
            ["examples/crack2.jpg"],
            ["examples/no_crack.jpg"]
        ],
        inputs = input_image
    )

    gr.Markdown("---")

    gr.Markdown(
        """
        👨‍💻 **Author:** 阎诗怡  

        🏫 **University:** NCWU University  

        🔗 **GitHub:**  
        https://github.com/12345678875/crack-detector
        """
    )

if __name__ == "__main__":
    demo.launch()
