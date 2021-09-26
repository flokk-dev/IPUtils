"""
Creator: Florian HOCQUET
Date: 26/09/2021
Version: 1.0

Purpose: Face Landmark on an image using mediapipe and OpenCV
"""

# LIB IMPORT
import numpy as np

import cv2 as cv
import mediapipe as mp


# CODE
class faceLandmark:
    """
    -----
    faceLandmark, a class to map someone's face.
    -----

    -----
    Attributes
    - _image : the image on which apply the face landmark.

    - _mpFaceMesh : mediapipe face landmarker.
    - _mpFaceMeshInstance : the instance of the mediapipe face landmarker.

    - _mpDrawing : mediapipe drawing object.
    - _mpDrawingStyles : styles for drawing object.
    - _mpDrawingSpec : specification for drawing object.
    -----

    -----
    Methods
    - process : Launch the face landmark process on an image.
    - _show : Show the result of the landmark process.
    -----

    """

    def __init__(self, nbFaces=1, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        """
        -----
        Initialise all the necessary attributes for the faceLandmark process.
        -----

        -----
        Parameters
        - :param nbFaces: the number of faces to landmark
          :type nbFaces: int

        - :param minDetectionConfidence: the minimum confidence value to get to validate the detection
          :type minDetectionConfidence: float

        - :param minTrackingConfidence: the minimum confidence value to get to validate the tracking
          :type minTrackingConfidence: float
        -----
        """
        self._image = None

        self._mpFaceMesh = mp.solutions.face_mesh
        self._mpFaceMeshInstance = self._mpFaceMesh.FaceMesh(max_num_faces=nbFaces,
                                                             min_detection_confidence=minDetectionConfidence,
                                                             min_tracking_confidence=minTrackingConfidence)

        self._mpDrawing = mp.solutions.drawing_utils
        self._mpDrawingStyles = mp.solutions.drawing_styles
        self._mpDrawingSpec = self._mpDrawing.DrawingSpec(thickness=1, circle_radius=1)

    def process(self, image, show=True):
        """
        -----
        Launch the faceLandmark process on an image.
        -----

        -----
        Parameters
        - :param image: the image on which apply the face landmark
          :type image: jpeg, jpg, png, numpy.ndarray...

        - :param show: if True, show the result of the landmark process
          :type show: bool
        -----

        -----
        Returns
        - :return shapes: the list containing a list of face's points for each person on an image
          :rtype shapes: list(list((int, int)))
        -----
        """
        if not isinstance(image, np.ndarray):
            image = cv.imread(image)

        self._image = image

        self._image = cv.cvtColor(cv.flip(self._image, 1), cv.COLOR_BGR2RGB)
        self._image.flags.writeable = False

        landmarkDatas = self._mpFaceMeshInstance.process(self._image)

        self._image.flags.writeable = True
        self._image = cv.cvtColor(self._image, cv.COLOR_RGB2BGR)

        if show and landmarkDatas.multi_face_landmarks:
            self._show(landmarkDatas)

        return landmarkDatas

    def _show(self, landmarkDatas):
        """
        -----
        Show the result of the landmark process
        -----

        -----
        Parameters
        - :param landmarkDatas: the list containing a list of face's points for each person on an image
          :type landmarkDatas: list(list((int, int)))
        -----
        """
        for face_landmarks in landmarkDatas.multi_face_landmarks:
            self._mpDrawing.draw_landmarks(image=self._image,
                                           landmark_list=face_landmarks,
                                           connections=self._mpFaceMesh.FACEMESH_TESSELATION,
                                           landmark_drawing_spec=self._mpDrawingSpec,
                                           connection_drawing_spec=self._mpDrawingSpec)

        cv.imshow('Landmark Detection', self._image)
