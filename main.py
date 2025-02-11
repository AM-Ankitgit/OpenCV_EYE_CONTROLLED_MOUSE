import cv2
import mediapipe as mp
import pyautogui

"""
MediaPipe is a comprehensive framework developed by Google for building and deploying machine learning pipelines, particularly for real-time processing of multimedia data like video and images.
It offers a wide range of pre-trained models and customizable components
"""

"""
PyAutoGUI is a cross-platform GUI automation Python module that allows users to create scripts to control the mouse and keyboard programmatically.
It can be used to automate tasks that involve interaction with a GUI, such as performing repetitive data entry tasks, testing software applications, and even bypassing systems that block automated browser use.
"""



# cap = cv2.VideoCapture(0)
# face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# screen_w,screen_h = pyautogui.size()

# while True:
#     isTrue,frame = cap.read()
#     frame = cv2.flip(frame,1) # 1 for vertical flip

#     rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     output = face_mesh.process(rgb_frame)
#     landmark_point = output.multi_face_landmarks
#     frame_h,frame_w ,_ = frame.shape
#     if landmark_point:
#         landmarks = landmark_point[0].landmark
#         for id,landmark in enumerate(landmarks[474:478]):
#             x = int(landmark.x*frame_w)
#             y = int(landmark.y*frame_h)
    

#             cv2.circle(frame,(x,y),3,(0,255,0))


#             if id==1:
#                 screen_x = screen_w/landmark.x
#                 screen_y = screen_w/landmark.y

#                 pyautogui.moveTo(screen_x,screen_y)

#         left = [landmarks[145],landmarks[159]]
#         for landmark in left:
#             x = int(landmark.x*frame_w)
#             y = int(landmark.y*frame_h)
    

#             cv2.circle(frame,(x,y),3,(0,255,255))

#         if (left[0].y - left[1].y) < 0.004:
#             pyautogui.click()
#             pyautogui.sleep(1)
            
#     cv2.imshow("frame",frame)
#     cv2.waitKey(1)



import cv2
import mediapipe as mp
import pyautogui
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)
    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)