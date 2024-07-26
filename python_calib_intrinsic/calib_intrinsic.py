import numpy as np
import cv2
from picamera2 import Picamera2

def calibrate(img_points, board_size, img_size):

    obj_points = np.zeros((board_size[0]*board_size[1], 3), dtype=np.float32)
    obj_points[:, :2] = np.mgrid[0:board_size[0],
                                 0:board_size[1]].T.reshape(-1, 2)
    obj_points = np.repeat(obj_points[None, :], len(img_points), axis=0)

    rms_err, \
        mat_cam, vec_dist, \
        rvecs, tvecs = \
            cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    return rms_err, mat_cam, vec_dist

def calib_intrinsic(img_size, board_size, fname_calib):

    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"format": "RGB888",
                                                               "size": img_size}))
    camera.start()

    corner_points = []
    while True:
        img = camera.capture_array()

        img_show = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(img_gray, board_size)
        if found:
            corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS +
                                        cv2.TERM_CRITERIA_MAX_ITER,
                                        30, 0.1))

            cv2.drawChessboardCorners(img_show, board_size, corners, found)

        cv2.imshow("img", img_show)

        key = cv2.waitKey(1)
        if key == ord(" "):
            if not found:
                continue

            cv2.putText(img_show, "Use this image (y)?", (15, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.imshow("img", img_show)
            key = cv2.waitKey()
            if key != ord("y"):
                continue

            corner_points += [np.squeeze(corners)]

        elif key == ord("c"):
            if len(corner_points) == 0:
                continue

            img_points = np.array(corner_points)

            rms_err, mat_cam, vec_dist = \
                calibrate(img_points, board_size, img_size)

            np.savez(fname_calib, rms_err=rms_err, mat_cam=mat_cam,
                     vec_dist=vec_dist)

            print(f"RMS error = {rms_err}")
            print(f"Mat Cam = {mat_cam}")
            print(f"Vec Dist = {vec_dist}")
            break

        elif key == 27:
            break

if __name__ == "__main__":
    img_size = (640, 480)
    board_size = (6, 5)
    fname_calib = "calib.npz"
    calib_intrinsic(img_size, board_size, fname_calib)

