import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import cv2

model_file = "lite0-detection-default.tflite"
#model_file = "lite0-detection-metadata.tflite"
#model_file = "lite0-int8.tflite"
#model_file = "ssd_mobilenet_default.tflite"

image_file = "grace_hopper.bmp"
label_file = "labelmap.txt"

interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
img = Image.open(image_file).resize((width, height))

# add N dim
input_data = np.expand_dims(img, axis=0)

# input_mean = 127.5
# input_std = 127.5
# if floating_model:
#     input_data = (np.float32(input_data) - input_mean) / input_std

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = []
for i in range(len(output_details)):
    output_data += [interpreter.get_tensor(output_details[i]['index'])]

with open(label_file, "r") as fp:
    labels = fp.readlines()
labels = [label.rstrip() for label in labels]

img = cv2.imread(image_file)

# output_data[0] format: [ymin, xmin, ymax, xmax]
pts = np.squeeze(output_data[0])
pts[:, 0::2] *= img.shape[0] # Top-Left point
pts[:, 1::2] *= img.shape[1] # Bot-Right point
pts = pts.astype(np.int64)

thresh = 0.3
for i in range(len(pts)):
    if output_data[2][0, i] < thresh:
        break

    tl = pts[i, :2][::-1]
    br = pts[i, 2:][::-1]
    cv2.rectangle(img, tl, br, (0, 255, 0))

    label = labels[int(output_data[1][0, i])]
    cv2.putText(img, f"{label} {output_data[2][0, i]:.2f}",
                tl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

cv2.imshow("img", img)
cv2.waitKey()
