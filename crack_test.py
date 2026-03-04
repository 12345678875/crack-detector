#
# import cv2
#
# img = cv2.imread("./test.png")  # 加个 ./ 表示当前目录
#
# if img is None:
#     print("foult")
# else:
#     print("succes")
#     print("size:" , img.shape)



# import cv2
#
# img = cv2.imread("./test.png")
#
# if img is None:
#     print("wrong")
#
# else:
#     print("ok")
#     print("size:" , img.shape)
#     cv2.imwrite("./output.png" , img)
#     print("已保存为output.png")
#
# cv2.imshow("crack image" , img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



 # 读取 灰度化 高斯滤波

import cv2

img = cv2.imread("./t5.png")

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
