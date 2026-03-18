#!/usr/bin/env python3
"""
STEP 3: Standalone ball detection using Kinova Gen3 built-in vision module.
Run this INDEPENDENTLY of the goalkeeper node to verify ball detection works.
No robot movement happens here — just camera + detection.

Requirements:
    pip3 install opencv-python numpy

Run:
    python3 ball_detection.py
"""

import cv2
import numpy as np
import os

# ── Camera stream ─────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.1.10"

# UDP avoids frame buffering buildup — important for real-time tracking
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Credentials and port included explicitly
RTSP_URL = f"rtsp://admin:admin@{ROBOT_IP}:554/color"
# ─────────────────────────────────────────────────────────────────────────────


# ── HSV color range ───────────────────────────────────────────────────────────
# Set TUNER_MODE = True first to find the right values for your ball + lighting.
# Then paste the values here and set TUNER_MODE = False.

TUNER_MODE = True   # <-- START here, tune first, then switch to False

BALL_HSV_LOWER = np.array([100, 80, 30])
BALL_HSV_UPPER = np.array([130, 255, 255])

# If using a RED ball instead, comment out the two lines above and use these:
# (red wraps around in HSV so needs two ranges — handled in detection below)
# USING_RED_BALL = True
# ─────────────────────────────────────────────────────────────────────────────

MIN_BALL_AREA = 500  # ignore blobs smaller than this (filters noise)


def open_camera():
    print(f"Connecting to {RTSP_URL} ...")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("ERROR: Could not open camera stream.")
        print("Troubleshooting:")
        print("  1. Make sure robot is on and Ethernet is connected")
        print("  2. ping 192.168.1.10")
        print("  3. Try opening in VLC: Media > Open Network Stream")
        print(f"     URL: {RTSP_URL}")
        return None

    # Read a test frame to confirm stream is live
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: Stream opened but could not read frame.")
        print("Try waiting a moment and running again.")
        cap.release()
        return None

    h, w = frame.shape[:2]
    print(f"Camera opened successfully! Resolution: {w}x{h}")
    return cap


def get_mask(hsv):
    """
    Returns a binary mask where the ball pixels are white.
    Uses BALL_HSV_LOWER/UPPER for yellow-green tennis ball by default.
    """
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)  # ← fixed: hsv not frame
    return mask


def run_hsv_tuner(cap):
    """
    Drag sliders until the ball shows as solid white in the mask window.
    Press 'q' to print final values.
    """
    print("\nHSV TUNER MODE")
    print("Adjust sliders until ONLY the ball appears WHITE in the mask window.")
    print("Everything else should be BLACK.")
    print("Press 'q' when done — final values will print to terminal.\n")

    def nothing(x):
        pass

    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H low",  "HSV Tuner", int(BALL_HSV_LOWER[0]), 179, nothing)
    cv2.createTrackbar("S low",  "HSV Tuner", int(BALL_HSV_LOWER[1]), 255, nothing)
    cv2.createTrackbar("V low",  "HSV Tuner", int(BALL_HSV_LOWER[2]), 255, nothing)
    cv2.createTrackbar("H high", "HSV Tuner", int(BALL_HSV_UPPER[0]), 179, nothing)
    cv2.createTrackbar("S high", "HSV Tuner", int(BALL_HSV_UPPER[1]), 255, nothing)
    cv2.createTrackbar("V high", "HSV Tuner", int(BALL_HSV_UPPER[2]), 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Lost camera frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lo = np.array([
            cv2.getTrackbarPos("H low",  "HSV Tuner"),
            cv2.getTrackbarPos("S low",  "HSV Tuner"),
            cv2.getTrackbarPos("V low",  "HSV Tuner"),
        ])
        hi = np.array([
            cv2.getTrackbarPos("H high", "HSV Tuner"),
            cv2.getTrackbarPos("S high", "HSV Tuner"),
            cv2.getTrackbarPos("V high", "HSV Tuner"),
        ])

        mask   = cv2.inRange(hsv, lo, hi)  # ← correct: filtering hsv
        result = cv2.bitwise_and(frame, frame, mask=mask)

        print(
            f"Lower: [{lo[0]:3d}, {lo[1]:3d}, {lo[2]:3d}]  "
            f"Upper: [{hi[0]:3d}, {hi[1]:3d}, {hi[2]:3d}]",
            end="\r"
        )

        cv2.imshow("Original", frame)
        cv2.imshow("Mask  (ball = WHITE, background = BLACK)", mask)
        cv2.imshow("Result (ball should be coloured here)", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n\nCopy these into your script:")
            print(f"BALL_HSV_LOWER = np.array([{lo[0]}, {lo[1]}, {lo[2]}])")
            print(f"BALL_HSV_UPPER = np.array([{hi[0]}, {hi[1]}, {hi[2]}])")
            print("Then set TUNER_MODE = False")
            break

    cv2.destroyAllWindows()


def run_detection(cap):
    """
    Detects ball and prints pixel coordinates in real time.
    """
    print("\nDETECTION MODE")
    print("Place tennis ball in camera view. Press 'q' to quit.")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Lost camera frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = get_mask(hsv)  # ← correct: passing hsv

        # Clean up mask
        mask = cv2.erode(mask,  None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ball_detected = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > MIN_BALL_AREA:
                (px, py), radius = cv2.minEnclosingCircle(largest)
                px, py = int(px), int(py)
                ball_detected = True

                print(f"Ball: pixel=({px:4d}, {py:4d})  "
                      f"radius={radius:5.1f}px  area={area:6.0f}", end="\r")

                cv2.circle(frame, (px, py), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"({px}, {py})",
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        if not ball_detected:
            print("No ball detected                              ", end="\r")
            cv2.putText(
                frame, "No ball detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )

        cv2.imshow("Ball Detection", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()
    cv2.destroyAllWindows()


def main():
    cap = open_camera()
    if cap is None:
        return

    if TUNER_MODE:
        run_hsv_tuner(cap)
    else:
        run_detection(cap)

    cap.release()


if __name__ == "__main__":
    main()
