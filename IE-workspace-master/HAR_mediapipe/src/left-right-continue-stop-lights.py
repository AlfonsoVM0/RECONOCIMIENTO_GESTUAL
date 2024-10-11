import cv2
import numpy as np
from keras.models import load_model, model_from_json
import mediapipe as mp
from landmarksLib import get_XYZ
from landmarksLib import get_XYZ, normalize_from_0_landmark

debug_HAR = False
only_landmarks = False
pred_mappings = ['continue','left','ligths','right','stop', 'NO_GESTURE']
landmark_values = [[0, 0] for _ in range(21)]
normalization = True
def main():
    large_size = (640, 480)
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # Usa la cámara por defecto de tu ordenador
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, large_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, large_size[1])

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Initialize Mediapipe
    print('Initializing Mediapipe')
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, 
                           max_num_hands=1, 
                           min_detection_confidence=0.5)

    # Load model if not only using landmarks
    if not only_landmarks:
        print("LOADING MODEL")

        weight_path = 'C:/Users/carlo/OneDrive - Universidad Politécnica de Madrid/Escritorio/Cuarto/PISD/IE-workspace-master/IE-workspace-master/IE-workspace-master/HAR_mediapipe/src/PIDS_CNN1_L0.h5'
        json_file = 'C:/Users/carlo/OneDrive - Universidad Politécnica de Madrid/Escritorio/Cuarto/PISD/IE-workspace-master/IE-workspace-master/IE-workspace-master/HAR_mediapipe/src/PIDS_CNN1_L0.json'

        with open(json_file) as f:
            loaded_model_json = f.read()

        model = model_from_json(loaded_model_json)
        model.load_weights(weight_path)

        print("LOADED MODEL:", json_file)

    print('Start processing frames')
    num_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        num_frames += 1

        if debug_HAR and num_frames % 25 == 0:
            print(f'num_frames = {num_frames}')

        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and get hand landmarks
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            if debug_HAR:
                print('Hand detected')

            image_rgb, landmark_values = get_XYZ(results, image_rgb)

        # Show the processed frame
        if only_landmarks:
            cv2.imshow("Hand Landmarks", image_rgb)
        else:
            if results.multi_hand_landmarks:
                landmarks_nparray = np.array(landmark_values)

                if normalization:
                    landmarks_nparray = np.array([element for sublist in landmarks_nparray for element in sublist])
                
                    # apply normalization
                    landmarks_nparray = normalize_from_0_landmark(landmarks_nparray)

                landmarks_nparray = np.reshape(landmarks_nparray, (1, 21, 2, 1))
                pred = pred_mappings[np.argmax(model(landmarks_nparray))]
                cv2.putText(image_rgb, 
                            "Predicted class: " + pred,
                            org=(10, large_size[1] - 20), 
                            fontFace=2,
                            fontScale=0.75,
                            color=(0, 200, 100))
            else:
                cv2.putText(image_rgb, 
                            "Predicted class: none",
                            org=(10, large_size[1] - 20), 
                            fontFace=2,
                            fontScale=0.75,
                            color=(0, 200, 100))
            cv2.imshow("Model prediction", image_rgb)

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
