# bilinear_src_x = (dst_x + 0.5) * (scr_width / maxWidth) - 0.5
# bilinear_src_y = (dst_y + 0.5) * (scr_height / maxHeight) - 0.5
import cv2
import numpy as np
import math

ori_image = cv2.imread('test2.jfif')
image = cv2.imread('test2.jfif')

bilinear_src_x_int = 2
bilinear_src_y_int = 2
u , v = 0.5, 0.5
Q11, Q12, Q21, Q22 = (
    (ori_image[bilinear_src_x_int][bilinear_src_y_int]),
    (ori_image[bilinear_src_x_int+1][bilinear_src_y_int]),
    (ori_image[bilinear_src_x_int][bilinear_src_y_int+1]),
    (ori_image[bilinear_src_x_int+1][bilinear_src_y_int+1]))

print(Q11,Q12, Q21, Q22)

dst_x = (Q11*(1-u)*(1-v)+
        Q12*(u)*(1-v)+
        Q21*(1-u)*(v)+
        Q22*(u)*(v))


print(dst_x)

print(image[100][100])
# cv2.imshow('Result', image[0][0]*2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# PD_A = np.mat([[0,maxHeight,0,1], [0,0,0,1], [maxWidth,0,0,1], [maxWidth,maxHeight,maxWidth*maxHeight,1]])
# PD_B = np.mat([ld, lt, rt, rd]).T
# a, b, c, d = np.linalg.solve(PD_A, PD_B).tolist()
#
# PD_C = np.mat([[0,maxHeight,0,1], [0,0,0,1], [maxWidth,0,0,1], [maxWidth,maxHeight,maxWidth*maxHeight,1]])
# PD_D = np.mat([ld, lt, rt, rd]).T
# e, f, g, h = np.linalg.solve(PD_C, PD_D).tolist()
# maxHeight=600
# maxWidth=300
# result = np.zeros((maxHeight,maxWidth,3), np.uint8)
#
# cv2.imshow('image',result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(result[200][100])
x = 3.14
y = math.floor(x)
print(type(y))
print(y+1)
