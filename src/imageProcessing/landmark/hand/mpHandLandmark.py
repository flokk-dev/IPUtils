"""
Creator: Florian HOCQUET
Date: 26/09/2021
Version: 1.0

Purpose: Hand Landmark on an image using mediapipe and OpenCV
"""

# LIB IMPORT
import numpy as np

import cv2 as cv
import mediapipe as mp

# PROJECT IMPORT
import src.imageProcessing.landmark.mediapipeLandmark as mpL


# CODE
class handLandmark(mpL.mediapipeLandmark):
    """
    -----
    handLandmark, a class to map someone's hand.
    -----

    -----
    Attributes
    - _image : the image on which apply the hand landmark.

    - _mpHand : mediapipe hand landmarker.
    - _mpHandInstance : the instance of the mediapipe hand landmarker.

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

    def __init__(self, nbElem=2, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        """
        -----
        Initialise all the necessary attributes for the faceLandmark process.
        -----

        -----
        Parameters
        - :param nbElem: the number of hands to landmark
          :type nbElem: int

        - :param minDetectionConfidence: the minimum confidence value to get to validate the detection
          :type minDetectionConfidence: float

        - :param minTrackingConfidence: the minimum confidence value to get to validate the tracking
          :type minTrackingConfidence: float
        -----
        """
        super().__init__()

        self._mpMesh = mp.solutions.hands
        self._mpMeshInstance = self._mpMesh.Hands(max_num_hands=nbElem,
                                                  min_detection_confidence=minDetectionConfidence,
                                                  min_tracking_confidence=minTrackingConfidence)

    def process(self, image, show=True):
        """
        -----
        Launch the mediapipeLandmark process on an image.
        -----

        -----
        Parameters
        - :param image: the image on which apply the body part landmark
          :type image: jpeg, jpg, png, numpy.ndarray...

        - :param show: if True, show the result of the landmark process
          :type show: bool
        -----

        -----
        Returns
        - :return shapes: the list containing a list of hand's points for each hand on an image
          :rtype shapes: list(list((int, int)))
        -----
        """
        landmarkDatas = super()._process(image, show)

        h, w = self._image.shape[:2]
        landmarkFinalDatas = {}

        if landmarkDatas.multi_hand_landmarks:
            for faceLandmarks in landmarkDatas.multi_hand_landmarks:
                for id, lm in enumerate(faceLandmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarkFinalDatas[id] = (x, y)

        return landmarkFinalDatas

    def _show(self, landmarkDatas):
        """
        -----
        Show the result of the landmark process
        -----

        -----
        Parameters
        - :param landmarkDatas: the list containing a list of hand's points for each person on an image
          :type landmarkDatas: list(list((int, int)))
        -----
        """
        if landmarkDatas.multi_hand_landmarks:
            for handLandmarks in landmarkDatas.multi_hand_landmarks:
                self._mpDrawing.draw_landmarks(image=self._image,
                                               landmark_list=handLandmarks,
                                               connections=self._mpMesh.HAND_CONNECTIONS,
                                               landmark_drawing_spec=self._mpDrawingSpec,
                                               connection_drawing_spec=self._mpDrawingSpec)

        cv.imshow('Landmark Detection', self._image)