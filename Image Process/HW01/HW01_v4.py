import cv2
import numpy as np
import math

four_points = []

#輸入Source Image, 與標記之四點, 利用Bilinear Interpolation得到校正後Image
def image_process(image, pts):
    ld, lt, rt, rd = pts
    #找出最大寬&高
    widthA = np.sqrt(((lt[0] - rt[0]) ** 2) + ((lt[1] - rt[1]) ** 2))
    widthB = np.sqrt(((rd[0] - ld[0]) ** 2) + ((rd[1] - ld[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((lt[0] - ld[0]) ** 2) + ((lt[1] - ld[1]) ** 2))
    heightB = np.sqrt(((rt[0] - rd[0]) ** 2) + ((rt[1] - rd[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 變形公式
    # x = ax' + by' + cx'y' + d
    # y = ex' + fy' + gx'y' + h
    # (x',y')為結果的點, (x,y)為對應至原圖的點, 代入4個點, 找出a,b,c,d,e,f,g
    # 可得到'結果圖之pixel' 對應至 '原圖上的pixel' -> 轉換方程式
    PD_A = np.mat([[0,maxHeight,0,1,0,0,0,0], [0,0,0,1,0,0,0,0], [maxWidth,0,0,1,0,0,0,0], [maxWidth,maxHeight,maxWidth*maxHeight,1,0,0,0,0],
                    [0,0,0,0,0,maxHeight,0,1], [0,0,0,0,0,0,0,1], [0,0,0,0,maxWidth,0,0,1], [0,0,0,0,maxWidth,maxHeight,maxWidth*maxHeight,1]])
    PD_B = np.mat([ld[0], lt[0], rt[0], rd[0],
                    ld[1], lt[1], rt[1], rd[1]]).T
    a, b, c, d, e, f, g, h = np.linalg.solve(PD_A, PD_B).tolist()

    #建立空白圖片
    Result = np.zeros((maxHeight,maxWidth,3), np.uint8)

    #將每個pixel找出對應pixel的(R,G,B)
    for x in range(maxWidth):   #寬
        for y in range(maxHeight):  #高
            bilinear_src_x = a[0]*x + b[0]*y + c[0]*x*y + d[0]  #結果x帶入公式，找出對應在原本圖的x上
            bilinear_src_y = e[0]*x + f[0]*y + g[0]*x*y + h[0]  #結果y帶入公式，找出對應在原本圖的y上

            #將浮點數(x,y)利用Bilinear Interpolation填入對應pixel(R,G,B)
            bilinear_src_x_int = math.floor(bilinear_src_x)
            bilinear_src_y_int = math.floor(bilinear_src_y)
            u = bilinear_src_x - bilinear_src_x_int
            v = bilinear_src_y - bilinear_src_y_int

            Q11, Q12, Q21, Q22 = (   #左上,右上,左下,右下
                (ori_image[bilinear_src_y_int][bilinear_src_x_int]),
                (ori_image[bilinear_src_y_int][bilinear_src_x_int+1]),
                (ori_image[bilinear_src_y_int+1][bilinear_src_x_int]),
                (ori_image[bilinear_src_y_int+1][bilinear_src_x_int+1]))
            Result[y][x] = (        #Bilinear Interpolation: 將附近4點利用權重算出(R,G,B),並將該pixel填入(R,G,B)
                Q11*(1-u)*(1-v)+
                Q12*(u)*(1-v)+
                Q21*(1-u)*(v)+
                Q22*(u)*(v))
    return Result

# if the left mouse button was clicked, then record to four_points[] & Mark Point in Red
def click_and_crop(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        four_points.append([x, y])
        # print(four_points)
        cv2.circle(image, (x,y), 3, (0,0,255), 5, 16)
        cv2.imshow("Select Corner", image)

#將四個點排列成 [左下, 左上, 右上, 右下]
def four_points_sort(pts):
    sort_x = pts[np.argsort(pts[:, 0]), :]
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    Left = Left[np.argsort(Left[:, 1])[::-1], :]
    Right = Right[np.argsort(Right[:, -1]), :]
    return np.concatenate((Left, Right), axis=0)

#Input image
ori_image = cv2.imread('input.png')
# print(ori_image.shape) #查看image size(Width, Height, RGB)
image = cv2.imread('input.png')

#Set Window to Record four_points[]
cv2.namedWindow("Select Corner")
cv2.setMouseCallback("Select Corner", click_and_crop)

# Find & Record four_points[]
while True:
    # display the image and wait for a keypress
    cv2.imshow("Select Corner", image)
    key = cv2.waitKey(1) & 0xFF
    if len(four_points) == 4:
        break
    # if the 'r' key is pressed, reset four_points
    elif key == ord("r"):
        image = ori_image.copy()
        four_points = []
    elif key == ord('x'):
        break

#Computing
four_points = four_points_sort(np.array(four_points, dtype=np.int32))   #重新排列為 four_points[左下, 左上, 右上, 右下]
# print('After sort:\n', four_points)
src_img_size = np.array(four_points, dtype="float32")
result = image_process(image, four_points)

#Show Result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
