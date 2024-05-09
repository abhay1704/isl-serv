import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# Initialize MediaPipe Drawing.

def detect_and_crop_hands(image):

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        raise Exception('No hands detected in the image')

    image_height, image_width, _ = image.shape

    cropped_images = []
    minX, minY, maxX, maxY = 0,0,10000,10000
    for hand_landmarks in results.multi_hand_landmarks:
        # Calculate the bounding box of the hand.
        min_x = min(landmark.x for landmark in hand_landmarks.landmark)
        max_x = max(landmark.x for landmark in hand_landmarks.landmark)
        min_y = min(landmark.y for landmark in hand_landmarks.landmark)
        max_y = max(landmark.y for landmark in hand_landmarks.landmark)

        # Convert the normalized coordinates to pixel coordinates with 10px padding.
        min_x = int(min_x * image_width - 10)
        max_x = int(max_x * image_width + 10)
        min_y = int(min_y * image_height - 10)
        max_y = int(max_y * image_height + 10)

        minX = min(minX, min_x)
        minY = min(minY, min_y)
        maxX = max(maxX, max_x)
        maxY = max(maxY, max_y)

        # Crop the image.
    cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image