import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Define deque arrays to handle color points for different colors
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Initialize color index for each color
color_indices = {'blue': 0, 'green': 0, 'red': 0, 'yellow': 0}

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Define colors and current color index
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Set up canvas
paintWindow = np.zeros((471, 636, 3)) + 255
buttons = {
    'CLEAR': (40, 1, 140, 65),
    'BLUE': (160, 1, 255, 65),
    'GREEN': (275, 1, 370, 65),
    'RED': (390, 1, 485, 65),
    'YELLOW': (505, 1, 600, 65)
}

for button, (x1, y1, x2, y2) in buttons.items():
    cv2.rectangle(paintWindow, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.putText(paintWindow, button, ((x1 + x2) // 2 - 30, (y1 + y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize MediaPipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame vertically and convert to RGB
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw buttons on the frame
    for button, (x1, y1, x2, y2) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, button, ((x1 + x2) // 2 - 30, (y1 + y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx, lmy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])

            # Draw landmarks
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = tuple(landmarks[8])
        thumb = tuple(landmarks[4])
        cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)

        if (thumb[1] - fore_finger[1]) < 30:
            for color in color_indices:
                color_indices[color] += 1
                locals()[f"{color[0]}points"].append(deque(maxlen=1024))

        elif fore_finger[1] <= 65:
            for button, (x1, y1, x2, y2) in buttons.items():
                if x1 <= fore_finger[0] <= x2:
                    if button == 'CLEAR':
                        for color in color_indices:
                            locals()[f"{color[0]}points"] = [deque(maxlen=1024)]
                            color_indices[color] = 0
                        paintWindow[67:, :, :] = 255
                    else:
                        colorIndex = list(buttons.keys()).index(button) - 1

        else:
            locals()[f"{colors[colorIndex][0] == 255 and 'b' or colors[colorIndex][1] == 255 and 'g' or colors[colorIndex][2] == 255 and 'r' or 'y'}points"][color_indices[list(color_indices.keys())[colorIndex]]].appendleft(fore_finger)

    else:
        for color in color_indices:
            color_indices[color] += 1
            locals()[f"{color[0]}points"].append(deque(maxlen=1024))

    # Draw lines on canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i, color_points in enumerate(points):
        for j in range(len(color_points)):
            for k in range(1, len(color_points[j])):
                if color_points[j][k - 1] is None or color_points[j][k] is None:
                    continue
                cv2.line(frame, color_points[j][k - 1], color_points[j][k], colors[i], 2)
                cv2.line(paintWindow, color_points[j][k - 1], color_points[j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
