import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
import numpy as np
import cv2
from PIL import Image
import datetime
from threading import Thread
from Spotipy import *
import time
import pandas as pd

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {0: "songs/angry.csv", 1: "songs/disgusted.csv", 2: "songs/fearful.csv",
              3: "songs/happy.csv", 4: "songs/neutral.csv", 5: "songs/sad.csv", 6: "songs/surprised.csv"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]


class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class VideoCamera:
    def __init__(self):
        self.df1 = pd.DataFrame([{
            'Id': 'N/A',
            'Name': 'No Data',
            'Album': 'N/A',
            'Artist': 'N/A',
            'Cover': 'https://via.placeholder.com/150'
        }])
        self.last_emotion = None
        self.last_emotion_time = time.time()
		
        
    def get_frame(self):
        global cap1
        cap1 = WebcamVideoStream(src=0).start()
        image = cap1.read()
        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        
        
        # Initialize df1 with default values to handle cases with no face detection

        if time.time() - self.last_emotion_time >= 0.1:
            self.last_emotion_time = time.time()
            if len(face_rects) > 0:
                for (x, y, w, h) in face_rects:
                    cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
                    roi_gray_frame = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                    prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                    if self.last_emotion != maxindex:
                        self.last_emotion = maxindex  # Update the last detected emotion
                        show_text[0] = maxindex
                        self.df1 = music_rec()  # Update song recommendations
                        print(f"Emotion changed to: {emotion_dict[maxindex]}")  # Debugging log
                    else:
                        print(f"Emotion unchanged: {emotion_dict[maxindex]}")
                    # show_text[0] = maxindex
                    cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # When no face is detected, show a message
                self.current_emotion = None
        
                self.df1 = pd.DataFrame([{
					'Id': 'N/A',
					'Name': 'No Data',
					'Album': 'N/A',
					'Artist': 'N/A',
					'Cover': 'https://via.placeholder.com/150'}])
                cv2.putText(image, "No face detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
			        # df1 = music_rec()  # Update df1 based on detected emotion
			


        global last_frame1
        last_frame1 = image.copy()
        img = Image.fromarray(last_frame1)
        img = np.array(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), self.df1


def music_rec():
    try:
        csv_path = music_dist.get(show_text[0], "songs/default.csv")
        df = pd.read_csv(csv_path)
        df = df[['Id', 'Name', 'Album', 'Artist', 'Cover']]
        df.insert(0, 'EmotionId', show_text[0])
        return df.head(15)
    except Exception as e:
        print(f"Error in music_rec: {e}")
        return pd.DataFrame([{
            'EmotionId': 'N/A',
            'Id': 'N/A',
            'Name': 'No Data',
            'Album': 'N/A',
            'Artist': 'N/A',
            'Cover': 'https://via.placeholder.com/150'
        }])


# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from pandastable import Table, TableModel
# import numpy as np
# import cv2
# from PIL import Image
# import datetime
# from threading import Thread
# from Spotipy import *
# import time
# import pandas as pd

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# ds_factor = 0.6

# emotion_model = Sequential()
# emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))
# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(7, activation='softmax'))
# emotion_model.load_weights('model.h5')

# cv2.ocl.setUseOpenCL(False)

# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# music_dist = {0: "songs/angry.csv", 1: "songs/angry.csv", 2: "songs/angry.csv",
#               3: "songs/happy.csv", 4: "songs/happy.csv", 5: "songs/sad.csv", 6: "songs/sad.csv"}

# global last_frame1
# last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
# global cap1
# show_text = [0]


# class FPS:
#     def __init__(self):
#         self._start = None
#         self._end = None
#         self._numFrames = 0

#     def start(self):
#         self._start = datetime.datetime.now()
#         return self

#     def stop(self):
#         self._end = datetime.datetime.now()

#     def update(self):
#         self._numFrames += 1

#     def elapsed(self):
#         return (self._end - self._start).total_seconds()

#     def fps(self):
#         return self._numFrames / self.elapsed()


# class WebcamVideoStream:
#     def __init__(self, src=0):
#         self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
#         (self.grabbed, self.frame) = self.stream.read()
#         self.stopped = False

#     def start(self):
#         Thread(target=self.update, args=()).start()
#         return self

#     def update(self):
#         while True:
#             if self.stopped:
#                 return
#             (self.grabbed, self.frame) = self.stream.read()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True


# class VideoCamera:
#     def __init__(self):
#         self.df1 = pd.DataFrame([{
#             'Id': 'N/A',
#             'Name': 'No Data',
#             'Album': 'N/A',
#             'Artist': 'N/A',
#             'Cover': 'https://via.placeholder.com/150'
#         }])
#         self.last_emotion = None
#         self.last_emotion_time = time.time()

#     def get_frame(self):
#         global cap1
#         cap1 = WebcamVideoStream(src=0).start()
#         image = cap1.read()
#         image = cv2.resize(image, (600, 500))
#         image = cv2.flip(image, 1)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

#         # Create a blurred version of the frame
#         blurred_frame = cv2.GaussianBlur(image, (51, 51), 0)

#         if time.time() - self.last_emotion_time >= 0.1:
#             self.last_emotion_time = time.time()
#             if len(face_rects) > 0:
#                 for (x, y, w, h) in face_rects:
#                     # Create a mask for the face region
#                     mask = np.zeros_like(image, dtype=np.uint8)
#                     cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

#                     # Invert the mask to target the background
#                     inverted_mask = cv2.bitwise_not(mask)

#                     # Keep the original face region
#                     face_region = cv2.bitwise_and(image, mask)

#                     # Apply the inverted mask to keep the blurred background
#                     background = cv2.bitwise_and(blurred_frame, inverted_mask)

#                     # Combine the face region with the blurred background
#                     image = cv2.add(face_region, background)

#                     roi_gray_frame = gray[y:y + h, x:x + w]
#                     cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#                     prediction = emotion_model.predict(cropped_img)
#                     maxindex = int(np.argmax(prediction))
#                     if self.last_emotion != maxindex:
#                         self.last_emotion = maxindex  # Update the last detected emotion
#                         show_text[0] = maxindex
#                         self.df1 = music_rec()  # Update song recommendations
#                         print(f"Emotion changed to: {emotion_dict[maxindex]}")  # Debugging log
#                     else:
#                         print(f"Emotion unchanged: {emotion_dict[maxindex]}")

#                     cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#             else:
#                 # When no face is detected, show a message
#                 self.current_emotion = None

#                 self.df1 = pd.DataFrame([{
#                     'Id': 'N/A',
#                     'Name': 'No Data',
#                     'Album': 'N/A',
#                     'Artist': 'N/A',
#                     'Cover': 'https://via.placeholder.com/150'}])
#                 cv2.putText(image, "No face detected", (20, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         global last_frame1
#         last_frame1 = image.copy()
#         img = Image.fromarray(last_frame1)
#         img = np.array(img)
#         ret, jpeg = cv2.imencode('.jpg', img)
#         return jpeg.tobytes(), self.df1


# def music_rec():
#     try:
#         csv_path = music_dist.get(show_text[0], "songs/default.csv")
#         df = pd.read_csv(csv_path)
#         df = df[['Id', 'Name', 'Album', 'Artist', 'Cover']]
#         df.insert(0, 'EmotionId', show_text[0])
#         return df.head(15)
#     except Exception as e:
#         print(f"Error in music_rec: {e}")
#         return pd.DataFrame([{
#             'EmotionId': 'N/A',
#             'Id': 'N/A',
#             'Name': 'No Data',
#             'Album': 'N/A',
#             'Artist': 'N/A',
#             'Cover': 'https://via.placeholder.com/150'
#         }])
