import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
img_size = 300
data_folder = "data/I love you"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
        img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = img_crop.shape
        # compare height and width to calculate the size of the (300px) box
        aspect_ratio = h / w

        # if height > width
        if aspect_ratio > 1:
            scaling_factor = img_size / h
            new_width = math.ceil(scaling_factor * w)
            resized_image = cv2.resize(img_crop, (new_width, img_size))
            resized_shape = resized_image.shape

            # center the image with white background
            width_gap = math.ceil((img_size - new_width) / 2)
            img_white[:, width_gap:new_width + width_gap] = resized_image

        # if width > height
        if aspect_ratio < 1:
            scaling_factor = img_size / w
            new_height = math.ceil(scaling_factor * h)
            resized_image = cv2.resize(img_crop, (img_size, new_height))
            resized_shape = resized_image.shape

            # center the image with white background
            height_gap = math.ceil((img_size - new_height) / 2)
            img_white[height_gap:new_height + height_gap, :] = resized_image

        cv2.imshow("Image Cropped", img_crop)
        cv2.imshow("White Background", img_white)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        time.sleep(0.1)
        counter += 1
        cv2.imwrite(f'{data_folder}/Image_{time.time()}.jpg', img_white)
        # count the number of pictures saved
        print(counter)

