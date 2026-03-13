import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load face detection model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load emotion model
emotion_model = load_model('emotion_model.hdf5')

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Start webcam
cap = cv2.VideoCapture(0)

# Setup real-time graph
plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(emotion_labels, [0]*7)
ax.set_ylim(0,1)
ax.set_title("Emotion Probability")
ax.set_ylabel("Confidence")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray,1.3,5,minSize=(30,30))

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(64,64))

        roi = roi_gray.astype('float')/255.0
        roi = np.reshape(roi,(1,64,64,1))

        prediction = emotion_model.predict(roi)

        max_index = np.argmax(prediction)
        label = emotion_labels[max_index]

        confidence = prediction[0][max_index]*100
        label_text = f"{label} ({confidence:.2f}%)"

        cv2.putText(frame,label_text,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)

        # Update graph
        for bar,val in zip(bars,prediction[0]):
            bar.set_height(val)

        fig.canvas.draw()
        fig.canvas.flush_events()

    cv2.imshow("Emotion Detection AI",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()