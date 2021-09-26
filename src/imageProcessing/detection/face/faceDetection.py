"""
Creator: Florian HOCQUET
Date: 19/09/2021
Version: 1.0

Purpose: Face Detection on image using dlib and OpenCV
"""

# PROJECT IMPORT
from paths import paths

# LIB IMPORT
import numpy as np
import cv2 as cv


# CODE
class faceDetection:
    """
    -----
    faceDetection, a class to detect someone's face.
    -----

    -----
    Attributes
    - model : serialized model.
    -----

    -----
    Methods
    - fd_process : Launch the faceDetection process on an image.
    - fd_blob : Generate a blob from an image.
    - fd_detection : Detect the different faces on an image.
    -----

    """

    def __init__(self):
        """ Initialise all the necessary attributes for the faceDetection process. """
        self._model = cv.dnn.readNetFromCaffe(paths.CAFFE_PROTO, paths.CAFEE_MODEL)

    def fd_process(self, image, minConfidence=0.7):
        """
        -----
        Launch the faceDetection process on an image.
        -----

        -----
        Parameters
        - :param image: the image on which you want to detect the faces
          :type image: jpeg, jpg, png...

        - :param minConfidence: the minimum confidence to validate a detection
          :type minConfidence: float
        -----

        -----
        Returns
        - :return faceCoordinates: the list containing tuples of face's coordinates for each person on an image
          :rtype faceCoordinates: list(list(np.int32))
        -----
        """
        self._fd_blob(image)

        image = cv.imread(image)
        (h, w) = image.shape[:2]

        return self._fd_detection(h, w, minConfidence)

    def _fd_blob(self, image):
        """
        -----
        Generate a blob from an image.
        -----

        -----
        Parameters
        - :param image: the image from which the blob will be generated
          :type image: jpeg, jpg, png...
        -----
        """
        blob = cv.dnn.blobFromImage(cv.resize(cv.imread(image), (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self._model.setInput(blob)

    def _fd_detection(self, h, w, minConfidence=0.7):
        """
        -----
        Detect the different faces on an image.
        -----

        -----
        Parameters
        - :param h: the height of the processed image
          :type h: int

        - :param w: the width of the processed image
          :type w: int

        - :param minConfidence: the minimum confidence to get to validate a face detection
          :type minConfidence: float
        -----

        -----
        Returns
        - :return faceCoordinates: the list containing tuples of face's coordinates for each person on an image
          :rtype faceCoordinates: list(list(np.int32))
        -----
        """
        detections = self._model.forward()

        faceCoordinates = list()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence >= minConfidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faceCoordinates.append(list(box.astype("int")))
            else:
                break

        return faceCoordinates
