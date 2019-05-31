# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="path to directory containing neural style transfer models")
args = vars(ap.parse_args())

model_paths = paths.list_files(args["models"], validExts=(".t7",))
model_paths = sorted(list(model_paths))

# Generate unique IDs for each of the model paths, then combine
models = list(zip(range(0, len(model_paths)), (model_paths)))

# Use the cycle function to loop over all models
model_iter = itertools.cycle(models)
(model_ID, model_path) = next(model_iter)

# Load the neural style transfer model
net = cv2.dnn.readNetFromTorch(model_path)

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
print("[INFO] {}. {}".format(model_ID + 1, model_path))

while True:
	frame = vs.read()

	# Resize the frame to have a width of 600 pixels
	frame = imutils.resize(frame, width=600)
	orig = frame.copy()
	(h, w) = frame.shape[:2]

	# Construct a blob from the frame and forward pass through network
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
		(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

    # Reshape the output tensor,
    # Add back in the mean subtraction
	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
    # Swap the channel ordering
	output = output.transpose(1, 2, 0)

	# show the original frame along with the output neural style
	# transfer
	cv2.imshow("Input", frame)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
