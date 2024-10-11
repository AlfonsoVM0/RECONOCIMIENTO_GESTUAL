import mediapipe as mp
import numpy as np

def get_XYZ(results, image_rgb):

    landmark_values = [[0, 0] for _ in range(21)]
    
    # esto para dibujar
    mp_drawing = mp.solutions.drawing_utils
    
    image_height, image_width, _ = image_rgb.shape
        
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(image_rgb, 
                                                  hand_landmarks, 
                                                  mp.solutions.hands.HAND_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=5),
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=4))
        
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x  # * image_width
            y = 1 - hand_landmarks.landmark[i].y  # * image_height (1-y for providing same shape as in image)
            landmark_values[i] = [x, y]
                
    return image_rgb, landmark_values

def normalize_from_0_landmark(data):
    # We create a copy of the input data array to ensure that the original data is not modified,
    # and the modifications are made to the copy
    new_data = np.copy(data)
    # This loop iterates through each row of the data array.
    # Each row represents a different hand instance or sample.

    # This condition helps to skip rows where the sum of X and Y coordinates is zero,
    # which may represent a case where there are no landmarks.
    if ((data[0] + data[1]) != 0):
        # These lines extract the X and Y coordinates of the center of the hand landmarks (typically, the palm).
        x_center = data[0]
        y_center = data[1]
        # This inner loop iterates through the remaining elements of the row,
        # starting from the third element (index 2).
        # In hand landmark data, these are typically the landmarks' X and Y coordinates, alternating.
        for k in range(2, data.shape[0], 2):
            # These lines calculate the new coordinates of each landmark relative to the center.
            # They subtract the center's X and Y coordinates from the corresponding landmark's X and Y coordinates.
            # This effectively shifts the coordinates so that the center becomes the new origin (0, 0).
            new_data[k] = data[k] - x_center
            new_data[k + 1] = data[k + 1] - y_center
    return new_data
