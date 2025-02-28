import os
import numpy as np
import cv2
from picamera2 import Picamera2

def calibrate(img_points, board_size, img_size, mat_cam=None, vec_dist=None,
              flags=None):

    obj_points = np.zeros((board_size[0]*board_size[1], 3), dtype=np.float32)
    obj_points[:, :2] = np.mgrid[0:board_size[0],
                                 0:board_size[1]].T.reshape(-1, 2)
    obj_points = np.repeat(obj_points[None, :], len(img_points), axis=0)

    rms_err, \
        mat_cam, vec_dist, \
        rvecs, tvecs = \
            cv2.calibrateCamera(obj_points, img_points, img_size,
                                mat_cam, vec_dist, flags=flags)

    return rms_err, mat_cam, vec_dist

def calib_intrinsic(camera, img_size, board_size, fname_calib=None,
                    mat_cam=None, vec_dist=None, flags=None,
                    file_prefix=None, do_calibration=True):
    """
    Intrinsic calibration helper function
    
    Procudure:
      1. Press space when you take the image
      2. Program asks if you use this image; then answer 'y' or else
      3. After enough images are captured press 'c' to calibrate

    if do_calibration:
      calibration is done
      if fname_calib is not None:
        calibration results are saved to file
      if file_prefix is not None:
        images and corner coordinates are saved to file
    else:
      calibration is not done
      if file_prefix is not None:
        images and corner coordinates are saved to file
    """
    rms_err = None
    
    if file_prefix is not None:
        board_imgs = []
        
    corner_points = []
    while True:
        img = camera.capture_array()

        img_show = img.copy()
        cv2.putText(img_show, f"{len(corner_points)}", (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow("img", img_show)

        key = cv2.waitKey(1)
        if key == ord(" "):

            found = False
            while not found:
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
                if cv2.waitKey(1) == 27:
                    break

            if found:
                cv2.putText(img_show, "Use this image (y)?", (15, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.imshow("img", img_show)
                key = cv2.waitKey()
                if key == ord("y"):
                    corner_points += [np.squeeze(corners)]
                    if file_prefix is not None:
                        board_imgs += [img]

        elif key == ord("c"):
            if len(corner_points) == 0:
                continue

            img_points = np.array(corner_points)

            if do_calibration:
                rms_err, mat_cam, vec_dist = \
                    calibrate(img_points, board_size, img_size, mat_cam, vec_dist,
                              flags)

                if fname_calib is not None:
                    ext = os.path.splitext(fname_calib)[1]
                    print(ext)
                    if ext == ".npz":
                        np.savez(fname_calib, rms_err=rms_err, mat_cam=mat_cam,
                                 vec_dist=vec_dist)
                    elif ext == ".yml":
                        fs = cv2.FileStorage(fname_calib, cv2.FileStorage_WRITE)
                        fs.write("MAT_CAM", mat_cam)
                        fs.write("VEC_DIST", vec_dist)
                        fs.write("RMS_ERR", rms_err)
                        fs.release()

                print(f"RMS error = {rms_err}")
                print(f"Mat Cam = {mat_cam}")
                print(f"Vec Dist = {vec_dist}")

            if file_prefix is not None:
                for i, img in enumerate(board_imgs):
                    fname_img = file_prefix + f"_img{i:02d}.png"
                    cv2.imwrite(fname_img, img)

                np.savez(file_prefix + "_img_points.npy", img_points)
            break

        elif key == 27:
            break

    return mat_cam, vec_dist, rms_err

if __name__ == "__main__":
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"format": "RGB888",
                                                               "size": img_size}))
    camera.start()

    img_size = (640, 480)
    board_size = (6, 5)
    fname_calib = "calib.npz"
    calib_intrinsic(camera, img_size, board_size, fname_calib)

