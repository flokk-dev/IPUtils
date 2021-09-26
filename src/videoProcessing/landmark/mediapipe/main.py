"""
Creator: Florian HOCQUET
Date: 26/09/2021
Version: 1.0

Purpose: script to use de mediapipe video landmark.
"""

# LIB IMPORT
import argparse

# PROJECT IMPORT
from src.videoProcessing.landmark.mediapipe import videoLandmark as vL

# CODE
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="apply landmark on the face, "
                                                    "following by the number of face to detect")
ap.add_argument("-ha", "--hand", required=True, help="apply landmark on the hand"
                                                     "following by the number of hand to detect")

args = ap.parse_args()

if args.face is not None and args.hand is not None:
    ap.error("you can't use two target for the landmark, choose only one of them")

elif args.face is not None and isinstance(args.face, int):
    videoLandmark = vL.videoLandmark("face", args.face)
    videoLandmark.process()

elif args.hand is not None and isinstance(args.hand, int):
    videoLandmark = vL.videoLandmark("hand", args.hand)
    videoLandmark.process()

else:
    ap.error("you need to use correctly the tools, read the help by using -h or --help")
