"""
Creator: Florian HOCQUET
Date: 19/09/2021
Version: 1.0

Purpose: Face Landmark on image using dlib and OpenCV.
"""

# PROJECT IMPORT
from paths import paths

# LIB IMPORT
import numpy as np
import cv2 as cv
import dlib


# CODE
class faceLandmark:
    """
    -----
    faceLandmark, a class to map someone's face.
    -----

    -----
    Attributes
    - detector : dlib's face detector.
    - predictor : facial landmark predictor.
    -----

    -----
    Methods
    - lm_process : Launch the landmark process on an image.
    - _lm_detection : Detect the different faces on an image.
    - _lm_prediction : Predict where the face's points are.

    -----

    """
    def __init__(self):
        """ Initialise all the necessary attributes for the faceLandmark process. """
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(paths.FACE_LANDMARK_68)

    def lm_process(self, image, single=True):
        """
        -----
        Launch the faceLandmark process on an image.
        -----

        -----
        Parameters
        - :param image: the image on which you want to landmark the faces
          :type image: jpeg, jpg, png, numpy.ndarray...

        - :param single: if True restrict the landmark to only one face
          :type single: bool
        -----

        -----
        Returns
        - :return shapes: the list containing a list of face's points for each person on an image
          :rtype shapes: list(list((int, int)))
        -----
        """
        if not isinstance(image, np.ndarray):
            image = cv.imread(image)

        gray, faces = self._lm_detection(image)
        return self._lm_prediction(faces, gray, single)

    def _lm_detection(self, image):
        """
        -----
        Detect the different faces on an image.
        -----

        -----
        Parameters
        - :param image: the image on which you want detect faces
          :type image: numpy.ndarray
        -----

        -----
        Returns
        - :returns gray: some color adjustments
          :rtype gray: numpy.ndarray

        - :returns faces: the faces's coordinates detected by the detector
          :rtype faces: list((int, int))
        -----
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        return gray, self._detector(gray, 1)

    def _lm_prediction(self, faces, gray, single):
        """
        -----
        Predict where the face's points are.
        -----

        -----
        Parameters
        - :param faces: the list of faces the predictor will process
          :type faces: list((int, int))

        - :param gray: some color adjustments
          :type gray: list((int, int))

        - :param single: if True restrict the prediction to only one face
          :type single: bool
        -----

        -----
        Returns
        - :return facesPoints: the list containing a list of face's points for each person on an image
          :rtype facesPoints: list(list((int, int)))
        -----
        """
        # Single face
        if single:
            facesPoints = np.zeros((68, 2), dtype="int")
            facePoints = self._predictor(gray, faces[0])

            for i in range(0, 68):
                facesPoints[i] = (facePoints.part(i).x, facePoints.part(i).y)

            return facesPoints

        # Several faces
        facesPoints = list()
        for face in faces:
            facesPoints_i = list()
            facePoints = self._predictor(gray, face)

            for i in range(0, 68):
                facesPoints_i.append((facePoints.part(i).x, facePoints.part(i).y))

            facesPoints.append(facesPoints_i)

        return facesPoints
