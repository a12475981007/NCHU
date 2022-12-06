import cv2
import numpy as np

four_points = []

#Define
def image_process(image, pts):
  ld, lt, rt, rd = pts
  widthA = np.sqrt(((lt[0] - rt[0]) ** 2) + ((lt[1] - rt[1]) ** 2))
  widthB = np.sqrt(((rd[0] - ld[0]) ** 2) + ((rd[1] - ld[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  heightA = np.sqrt(((lt[0] - ld[0]) ** 2) + ((lt[1] - ld[1]) ** 2))
  heightB = np.sqrt(((rt[0] - rd[0]) ** 2) + ((rt[1] - rd[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  #Set Result Size
  dst_image_size = np.array([
    [0, maxHeight],
    [0, 0],
    [maxWidth, 0],
    [maxWidth, maxHeight]], dtype="float32")
  M = cv2.getPerspectiveTransform(src_img_size, dst_image_size)
  Result = cv2.warpPerspective(ori_image, M, (maxWidth, maxHeight))
  return Result

def click_and_crop(event, x, y, flags, param):
    # if the left mouse button was clicked, record to four_points, and Mark Point
    if event == cv2.EVENT_LBUTTONDOWN:
        four_points.append([x, y])
        print(four_points)
        cv2.circle(image, (x,y), 3, (0,0,255), 5, 16)
        cv2.imshow("Select Corner", image)

def four_points_sort(pts):
    sort_x = pts[np.argsort(pts[:, 0]), :]
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    Left = Left[np.argsort(Left[:, 1])[::-1], :]
    Right = Right[np.argsort(Right[:, -1]), :]
    return np.concatenate((Left, Right), axis=0)

#Input
ori_image = cv2.imread('input.png')
image = cv2.imread('input.png')

cv2.namedWindow("Select Corner")
cv2.setMouseCallback("Select Corner", click_and_crop)

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
four_points = four_points_sort(np.array(four_points, dtype=np.int32))
print(four_points)
src_img_size = np.array(four_points, dtype="float32")
result = image_process(image, four_points)

#Show Result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
