# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="path to directory containing neural style transfer models")
args = vars(ap.parse_args())

model_paths = paths.list_files(args["models"], validExts=(".t7",))
model_paths = sorted(list(model_paths))

# generate unique IDs for each of the model paths, then combine
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

# use the cycle function to loop over all models
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)
