import os
import argparse
import numpy as np
import cv2
from picamera2 import Picamera2

class CalibParam:
    def __init__(self, fname_config):
        if fname_config is None:
            self.init_mat_cam = None
            self.init_vec_dist = None
            self.flags = 0
            return

        fs = cv2.FileStorage(fname_config, cv2.FileStorage_READ)
        self.init_mat_cam = fs.getNode("INIT_MAT_CAM").mat()
        self.init_vec_dist = fs.getNode("INIT_VEC_DIST").mat()
        self.flags = 0
        self.flags += int(fs.getNode("USE_INTRINSIC_GUESS").real())     * 0x00001
        self.flags += int(fs.getNode("USE_FIX_ASPECT_RATIO").real())    * 0x00002
        self.flags += int(fs.getNode("USE_FIX_PRINCIPAL_POINT").real()) * 0x00004
        self.flags += int(fs.getNode("ZERO_TANGENT_DIST").real())       * 0x00008
        self.flags += int(fs.getNode("FIX_FOCAL_LENGTH").real())        * 0x00010
        self.flags += int(fs.getNode("FIX_K1").real())                  * 0x00020
        self.flags += int(fs.getNode("FIX_K2").real())                  * 0x00040
        self.flags += int(fs.getNode("FIX_K3").real())                  * 0x00080
        self.flags += int(fs.getNode("FIX_K4").real())                  * 0x00800
        self.flags += int(fs.getNode("FIX_K5").real())                  * 0x01000
        self.flags += int(fs.getNode("FIX_K6").real())                  * 0x02000
        self.flags += int(fs.getNode("RATIONAL_MODEL").real())          * 0x04000
        self.flags += int(fs.getNode("THIN_PRISM_MODEL").real())        * 0x08000
        self.flags += int(fs.getNode("FIX_S1_S2_S3_S4").real())         * 0x10000
        self.flags += int(fs.getNode("TILTED_MODEL").real())            * 0x40000
        self.flags += int(fs.getNode("FIX_TAUX_TAUY").real())           * 0x80000
        self.flags += int(fs.getNode("FIX_TANGENT_DIST").real())       * 0x200000
        fs.release()

def calibrate(img_points, board_size, img_size, calib_param=None):

    obj_points = np.zeros((board_size[0]*board_size[1], 3), dtype=np.float32)
    obj_points[:, :2] = np.mgrid[0:board_size[0],
                                 0:board_size[1]].T.reshape(-1, 2)
    obj_points = np.repeat(obj_points[None, :], len(img_points), axis=0)

    rms_err, \
        mat_cam, vec_dist, \
        rvecs, tvecs = \
            cv2.calibrateCamera(obj_points, img_points, img_size,
                                calib_param.init_mat_cam,
                                calib_param.init_vec_dist,
                                flags=calib_param.flags)

    print(calib_param.flags)

    return rms_err, mat_cam, vec_dist

def calib_intrinsic(camera, img_size, board_size, fname_calib=None, fname_config=None,
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
        cv2.putText(img_show, f"{len(corner_points)} images taken.", (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.putText(img_show, "Press space to take image.", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.putText(img_show, "Press 'c' to calibrate.", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow("img", img_show)

        key = cv2.waitKey(1)
        if key == ord(" "):

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(img_gray, board_size)
            if found:
                corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS +
                                            cv2.TERM_CRITERIA_MAX_ITER,
                                            30, 0.1))

                img_show = img.copy()
                cv2.drawChessboardCorners(img_show, board_size, corners, found)
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
                calib_param = CalibParam(fname_config)

                rms_err, mat_cam, vec_dist = \
                    calibrate(img_points, board_size, img_size, calib_param)

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
                    fname_img = file_prefix + f"_img{i:02d}.jpg"
                    cv2.imwrite(fname_img, img)

                np.savez(file_prefix + "_img_points", img_points)
            break

        elif key == 27:
            break

    return mat_cam, vec_dist, rms_err

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Camera calibration software")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--corners_h", type=int, default=6)
    parser.add_argument("--corners_v", type=int, default=5)
    parser.add_argument("--dname_results", type=str, default="results")
    parser.add_argument("--fname_calib", type=str, default="calib.yml")
    parser.add_argument("--fname_config", type=str, default="config.yml")
    parser.add_argument("--out_prefix", type=str, default="out")

    args = parser.parse_args()
    
    img_size = (args.width, args.height)
    board_size = (args.corners_h, args.corners_v)
    dname_results = args.dname_results
    fname_calib = os.path.join(dname_results, args.fname_calib)
    out_file_prefix = os.path.join(dname_results, args.out_prefix)
    fname_config = args.fname_config

    os.makedirs(dname_results, exist_ok=True)

    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"format": "RGB888",
                                                               "size": img_size}))
    camera.start()
    
    calib_intrinsic(camera, img_size, board_size, fname_calib, fname_config,
                    out_file_prefix)

