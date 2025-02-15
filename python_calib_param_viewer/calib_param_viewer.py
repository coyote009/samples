import cv2
import numpy as np

class CalibParam:
    def __init__(self, slide_init, slide_max, min_val, max_val):
        self.slide_pos = slide_init
        self.slide_max = slide_max
        self.min_val = min_val
        self.max_val = max_val
        self.val = self.compute_val()

    def compute_val(self):
        return self.min_val + self.slide_pos/self.slide_max * (self.max_val - \
                                                                self.min_val)

    def on_trackbar(self, slide_pos):
        if slide_pos < 0:
            self.slide_pos = 0
        elif slide_pos > self.slide_max:
            self.slide_pos = self.max_val
        else:
            self.slide_pos = slide_pos

        self.val = self.compute_val()

# Output image size
img_size = (640, 480)

# Camera Parameters
param_f = CalibParam(50, 100, 600, 680)
param_cu = CalibParam(50, 100, 300, 340)
param_cv = CalibParam(50, 100, 220, 260)
param_k1 = CalibParam(50, 100, -2.0, 2.0)
param_k2 = CalibParam(50, 100, -2.0, 2.0)
param_p1 = CalibParam(50, 100, -0.5, 0.5)
param_p2 = CalibParam(50, 100, -0.5, 0.5)
param_s1 = CalibParam(50, 100, -0.5, 0.5)
param_s3 = CalibParam(50, 100, -0.5, 0.5)
param_tx = CalibParam(50, 100, -0.5, 0.5)
param_ty = CalibParam(50, 100, -0.5, 0.5)

cv2.namedWindow("img")

cv2.createTrackbar('f ', "img", param_f.slide_pos, param_f.slide_max,
                   param_f.on_trackbar)
cv2.createTrackbar('cu', "img", param_cu.slide_pos, param_cu.slide_max,
                   param_cu.on_trackbar)
cv2.createTrackbar('cv', "img", param_cv.slide_pos, param_cv.slide_max,
                   param_cv.on_trackbar)
cv2.createTrackbar('k1', "img", param_k1.slide_pos, param_k1.slide_max,
                   param_k1.on_trackbar)
cv2.createTrackbar('k2', "img", param_k2.slide_pos, param_k2.slide_max,
                   param_k2.on_trackbar)
cv2.createTrackbar('p1', "img", param_p1.slide_pos, param_p1.slide_max,
                   param_p1.on_trackbar)
cv2.createTrackbar('p2', "img", param_p2.slide_pos, param_p2.slide_max,
                   param_p2.on_trackbar)
cv2.createTrackbar('s1', "img", param_s1.slide_pos, param_s1.slide_max,
                   param_s1.on_trackbar)
cv2.createTrackbar('s3', "img", param_s3.slide_pos, param_s3.slide_max,
                   param_s3.on_trackbar)
cv2.createTrackbar('tx', "img", param_tx.slide_pos, param_tx.slide_max,
                   param_tx.on_trackbar)
cv2.createTrackbar('ty', "img", param_ty.slide_pos, param_ty.slide_max,
                   param_ty.on_trackbar)

# Generate original chart image
img_size_chart = (1280, 960)
img_chart = np.zeros(img_size_chart[::-1], np.uint8)
for j in range(6):
    for i in range(8):
        if (i%2 and j%2) or (not i%2 and not j%2):
            img_chart = cv2.rectangle(img_chart,
                                      (640-100*4+100*i, 480-100*3+100*j),
                                      (640-100*4+100*(i+1)-1, 480-100*3+100*(j+1)-1),
                                      (255, 255, 255), cv2.FILLED)

mat_cam_chart = np.array([[1280., 0., 640.],
                          [0., 1280., 480.],
                          [0., 0., 1.]])

while(True):
    mat_cam = np.array([[param_f.val, 0, param_cu.val],
                        [0, param_f.val, param_cv.val],
                        [0, 0, 1]])
    vec_dist = np.array([param_k1.val, param_k2.val, param_p1.val, param_p2.val,
                         0., 0., 0., 0., param_s1.val, 0., param_s3.val, 0.,
                         param_tx.val, param_ty.val])

    mapx, mapy = cv2.initInverseRectificationMap(mat_cam, vec_dist, np.eye(3),
                                                 mat_cam_chart, img_size,
                                                 cv2.CV_32FC1)
    img = cv2.remap(img_chart, mapx, mapy, cv2.INTER_LANCZOS4)

    cv2.imshow("img", img)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
