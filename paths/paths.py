"""
Creator: Florian HOCQUET
Date: 18/09/2021
Version: 1.0

Purpose: Store all the path constant in a single file
"""


# LIB IMPORT
import os


# PATHS
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESOURCES_PATH = os.path.join(ROOT_PATH, "resources")

IMAGES_PATH = os.path.join(RESOURCES_PATH, "images" + os.sep)

COVID_MASK = os.path.join(IMAGES_PATH, "covidMask.png")

FACE_0 = os.path.join(IMAGES_PATH, "face0.jpeg")

FACES_0 = os.path.join(IMAGES_PATH, "faces0.jpg")

SHAPE_PREDICTOR_PATH = os.path.join(RESOURCES_PATH, "models" + os.sep)

FACE_LANDMARK_68 = os.path.join(SHAPE_PREDICTOR_PATH, "faceLandmark68.dat")
CAFFE_PROTO = os.path.join(SHAPE_PREDICTOR_PATH, "prototxt.txt")
CAFEE_MODEL = os.path.join(SHAPE_PREDICTOR_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
