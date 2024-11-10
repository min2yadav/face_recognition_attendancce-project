import cv2
import pickle
import numpy as np
import os
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data=[]

i=0

name=input("Enter Your Name: ")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==100:
        break
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100, -1)


if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
        
        
 #==================================using trainded DL model=====================================          
        
        
import cv2
import pickle
import numpy as np
import os
import torch  # Example for PyTorch-based model
from your_model_file import YourFaceDetectionModel  # Import your trained face detection model

# Load the trained model
model = YourFaceDetectionModel()
model.load_state_dict(torch.load("your_trained_model.pt"))  # Load weights
model.eval()

video = cv2.VideoCapture(0)

faces_data = []
name = input("Enter Your Name: ")

# Capture and process frames from webcam
while True:
    ret, frame = video.read()

    # Preprocess the frame for your model
    img_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255  # Model-specific input format
    with torch.no_grad():
        outputs = model(img_tensor)  # Get detections (bounding boxes, scores, etc.)
    
    # Assuming the model returns bounding boxes and associated scores
    boxes = outputs[0]  # Get bounding boxes from your model output
    scores = outputs[1]  # Get confidence scores for each box
    
    # Process the detections
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score > 0.5:  # Threshold score to filter weak detections
            x, y, w, h = box  # Assuming box is in format [x, y, width, height]
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))  # Resize to model input size
            faces_data.append(resized_img)

            cv2.putText(frame, f"Captured {len(faces_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= 100:
        break

video.release()
cv2.destroyAllWindows()

# Convert the captured faces data to numpy array and save
faces_data = np.vstack(faces_data)  # Shape (100, embedding_dim)

# Save the new embeddings with a name label
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * len(faces_data)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * len(faces_data)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Save or append face embeddings data
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        existing_faces = pickle.load(f)
    existing_faces = np.vstack((existing_faces, faces_data))
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(existing_faces, f)
        