"""
Creator: Florian HOCQUET
Date: 26/09/2021
Version: 1.0

Purpose: Landmark on an image using mediapipe and OpenCV
"""

# LIB IMPORT
import numpy as np

import cv2 as cv
import mediapipe as mp


# CODE
class mediapipeLandmark:
    """
    -----
    mediapipeLandmark, a class to map someone's body part.
    -----

    -----
    Attributes
    - _image : the image on which apply the body part landmark.

    - _mpMesh : mediapipe body part landmarker.
    - _mpMeshInstance : the instance of the mediapipe body part landmarker.

    - _mpDrawing : mediapipe drawing object.
    - _mpDrawingStyles : styles for drawing object.
    - _mpDrawingSpec : specification for drawing object.
    -----

    -----
    Methods
    - _process : Launch the face landmark process on an image.
    -----

    """

    def __init__(self):
        """
        -----
        Initialise all the necessary attributes for the mediapipeLandmark process.
        -----
        """
        self._image = None

        self._mpDrawing = mp.solutions.drawing_utils
        self._mpDrawingStyles = mp.solutions.drawing_styles
        self._mpDrawingSpec = self._mpDrawing.DrawingSpec(thickness=1, circle_radius=1)

    def _process(self, image, show=True):
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
        - :return shapes: the list containing a list of body part's points for each person on an image
          :rtype shapes: mediapipe.SolutionOutputs
        -----
        """
        if not isinstance(image, np.ndarray):
            image = cv.imread(image)

        self._image = image

        self._image = cv.cvtColor(cv.flip(self._image, 1), cv.COLOR_BGR2RGB)
        self._image.flags.writeable = False

        landmarkDatas = self._mpMeshInstance.process(self._image)

        self._image.flags.writeable = True
        self._image = cv.cvtColor(self._image, cv.COLOR_RGB2BGR)

        if show:
            self._show(landmarkDatas)

        return landmarkDatas
