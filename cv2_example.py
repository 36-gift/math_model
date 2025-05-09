# 几个常用函数：图像读取——cv2.imread()，图像显示——cv2.imshow()，
# 图像保存——cv2.imwrite()，图像关闭——cv2.destroyAllWindows()
# 图像转化——cv2.cvtColor(),图像灰度化——cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),边缘检测——cv2.Canny()

#下面三行是copliot给出的函数，不一定常用
# 图像滤波——cv2.blur(),cv2.GaussianBlur(),cv2.medianBlur(),cv2.bilateralFilter()
# 图像形态学——cv2.erode(),cv2.dilate(),cv2.morphologyEx()
# 图像特征提取——cv2.HoughCircles(),cv2.HoughLines()



#1）：读取和显示图像
import cv2
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread(r'D:\python_application\ONNX\test1.png') #替换成具体路径,前面加上r表示字符串不转义

# 将BGR图像转为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用matplotlib显示图像
plt.imshow(img_rgb)
plt.axis('off')  # 不显示坐标轴
plt.show()

#2）：对图像进行简单的灰度转换和边缘检测
import cv2

# 加载图像
img = cv2.imread(r'D:\python_application\ONNX\test1.png', 0)  # 0表示以灰度模式读取

# 应用Canny边缘检测
edges = cv2.Canny(img, 100, 200)

# 显示结果
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


#3)：使用OpenCV的Hough变换检测直线
import cv2
import numpy as np
# 加载图像
img = cv2.imread(r'D:\python_application\ONNX\test1.png')
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 应用Canny边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# 应用Hough变换检测直线
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# 绘制检测到的直线
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
cv2.imshow('Hough Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#4)：使用OpenCV的Hough变换检测圆
import cv2
import numpy as np
# 加载图像
img = cv2.imread(r'D:\python_application\ONNX\test1.png', 0)
# 应用Hough变换检测圆
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
# 绘制检测到的圆
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # 绘制外圆
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 绘制圆心
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

# 显示结果
cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#5)：使用OpenCV的Hough变换检测椭圆
import cv2
# 加载图像
img = cv2.imread(r'D:\python_application\ONNX\test1.png', 0)
# 应用Hough变换检测椭圆
ellipses = cv2.fitEllipse(img)
# 绘制检测到的椭圆
cv2.ellipse(img, ellipses, (0, 255, 0), 2)
# 显示结果
cv2.imshow('detected ellipse', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#6)：使用OpenCV的Hough变换检测矩形
import cv2
# 加载图像
img = cv2.imread(r'D:\python_application\ONNX\test1.png')
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 应用Canny边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# 应用Hough变换检测直线
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# 绘制检测到的直线
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
cv2.imshow('Hough Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#7)：使用OpenCV的Hough变换检测多边形
import cv2
# 加载图像
img = cv2.imread(r'D:\python_application\ONNX\test1.png')
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 应用Canny边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# 应用Hough变换检测直线
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# 绘制检测到的直线
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
cv2.imshow('Hough Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#8)：使用OpenCV的Hough变换检测任意形状
import cv2
import numpy as np
# 加载图像
img = cv2.imread(r'D:\python_application\ONNX\test1.png')
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 应用Canny边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# 应用Hough变换检测直线
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# 绘制检测到的直线
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
cv2.imshow('Hough Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()