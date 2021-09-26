"""
Creator: Florian HOCQUET
Date: 26/09/2021
Version: 1.0

Purpose: Landmark on a video stream using mediapipe and OpenCV.
"""

# LIB IMPORT
import mediapipe as mp
import cv2 as cv

# PROJECT IMPORT
import src.imageProcessing.landmark.face.mpFaceLandmark as fL
import src.imageProcessing.landmark.hand.mpHandLandmark as hL


# CODE
class videoLandmark:
    """
    -----
    videoLandmark, a class to landmark someone's part of the body.
    -----

    -----
    Attributes
    - _videoStream : the video stream on which we apply landmark process.
    - _imageProcessing : the imageProcessing corresponding to the bodyPart to landmark.

    - _mpDrawing : mediapipe drawing object.
    - _mpDrawingStyles : styles for drawing object.
    - _mpDrawingSpec : specification for drawing object.
    -----

    -----
    Methods
    - process : Launch the landmark process on a videoStream.
    -----

    """

    _PROCESSING = {
        "face": fL.faceLandmark,
        "hand": hL.handLandmark
    }

    def __init__(self, bodyPart, nbElem=1, minDetectionConfidence=0.6, minTrackingConfidence=0.6):
        """
        -----
        Initialise all the necessary attributes for the faceLandmark process.
        -----

        -----
        Parameters
        - :param bodyPart: the body part to landmark
          :type bodyPart: str

        - :param nbElem: the number of elements to landmark
          :type nbElem: int

        - :param minDetectionConfidence: the minimum confidence value to get to validate the detection
          :type minDetectionConfidence: float

        - :param minTrackingConfidence: the minimum confidence value to get to validate the tracking
          :type minTrackingConfidence: float
        -----
        """
        if bodyPart is None:
            raise ValueError("The program needs to know which part of the body it has to landmark to work.")

        if not isinstance(bodyPart, str):
            raise TypeError("The bodyPart argument needs to be a str")

        self._videoStream = None

        self._imageProcessing = self._PROCESSING[bodyPart](nbElem=nbElem,
                                                           minDetectionConfidence=minDetectionConfidence,
                                                           minTrackingConfidence=minTrackingConfidence)

    def process(self, videoStream=0):
        """
        -----
        Launch the landmark process on an image.
        -----

        -----
        Parameters
        - :param videoStream: the id of the videoStream to use as the input
          :type videoStream: int
        -----

        -----
        Returns
        - :return shapes: the list containing a list of face's points for each person on an image
          :rtype shapes: list(list((int, int)))
        -----
        """
        if not isinstance(videoStream, int):
            raise TypeError("The videoStream argument needs to be a int")

        self._videoStream = cv.VideoCapture(0)

        while self._videoStream.isOpened():
            success, image = self._videoStream.read()
            if not success:
                continue

            landmarkDatas = self._imageProcessing.process(image)
            print(landmarkDatas)

            if cv.waitKey(10) == 27:
                self._videoStream.release()
