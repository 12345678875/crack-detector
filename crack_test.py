# 读取 灰度化 高斯滤波
import cv2

img = cv2.imread("image/t5.png")

# 灰度化
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

print("原图尺寸：" , img.shape)
print("灰度图尺寸：" , gray.shape)

cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存灰度图
cv2.imwrite("./gray.png", gray)

# 高斯滤波

blurred = cv2.GaussianBlur(gray , (5,5) , 0)

cv2.imshow("Blurred image" , blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./blurred.png" , blurred)

# 阈值二值化
_,binary = cv2.threshold(blurred , 150,255 , cv2.THRESH_BINARY)

cv2.imshow("Binary image" , binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite("./binary.png" , binary)


edges = cv2.Canny(blurred , 80 , 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./edges.png", edges)
print("边缘检测完成")

# import cv2
#
# # 读取 → 灰度 → 滤波
# img = cv2.imread("./t5.png")#读取
# gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)#//灰度化
# blurred = cv2.GaussianBlur(gray , (5 , 5) , 0)#//高斯滤波
# _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)#//阈值
#
# # Canny 边缘检测
# edges = cv2.Canny(blurred , 80 , 200)
#
# cv2.imshow("Edges", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imwrite("./edges.png", edges)
# print("边缘检测完成")
import cv2
import numpy as np
import datetime
import os
import glob

# # ============ 配置参数 ============
PIXELS_PER_MM = 12
MIN_AREA = 100

# ============ 找到所有图片 ============
# 直接读取当前目录下的所有 png 图片
# image_files = glob.glob("*.png")
#读取image里面的图片
image_files = glob.glob("image/*.png")
# 过滤掉结果图（避免重复处理）
image_files = [f for f in image_files if not f.endswith("_result.png")]

print(f"找到 {len(image_files)} 张图片")
print("=" * 50)

all_reports = []

for img_path in image_files:
    print(f"\n处理：{img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print("  跳过：无法读取")
        continue


# 读取 → 灰度 → 滤波 → 边缘检测
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray , (5,5) , 0)
    edges = cv2.Canny(blurred , 80 , 200)

# 轮廓提取
    contours , _ = cv2.findContours(edges , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    print("检测到" , len(contours) , "个轮廓")

# 在原图上画出轮廓
    img_copy = img.copy()
    cv2.drawContours(img_copy , contours , -1 , (0 , 255 , 0) , 2)

    cv2.imshow("Contours" , img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("./countours.png" , img_copy)

    crack_count = 0

# # 生成报告内容
    report = []
    report.append("="*50)
    report.append("裂缝检测报告")
    report.append(f"检测时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"图片文件：t5.png")
    report.append("="*50)
    report.append("")

    crack_count = 0

    for i , contour in enumerate(contours):
        length = cv2.arcLength(contour , True)
        area = cv2.contourArea(contour)
        if area > 100:
             print(f"轮廓{i}: 长度={length:.2f}像素，面积={area:.2f}像素²")
    if area > MIN_AREA:
        length_mm = length / PIXELS_PER_MM
        width = area / (length / 2)
        width_mm = width / PIXELS_PER_MM

        report.append(f"裂缝{crack_count}:")
        report.append(f"  长度：{length_mm:.2f}mm({length:.2f}像素)")
        report.append(f"  平均宽度：{width_mm:.2f}mm({width:.2f}像素)")
        report.append(f"  面积:{area:.2f}像素")

        crack_count += 1

    report.append("="*50)
    report.append(f"共检测到{crack_count}条有效裂缝")

##保存到文件

    report_text = "\n".join(report)
# # 保存标注图片
    all_reports.append(report_text)
    output_path = img_path.replace(".png", "_result.png")
    cv2.imwrite(output_path , img_copy)
    print(f"  结果已保存：{output_path}")
    print(f"  检测到{crack_count} 条裂缝")

with open("./batch_report.txt" , "w" , encoding = "utf-8") as f:
    f.write("\n".join(all_reports))

print("\n" + " = " * 50)
print("批量处理完成！")
print("总报告已保存：batch_report.txt")

