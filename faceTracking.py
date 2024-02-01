import numpy as np
import cv2
import face_recognition as fr


# load a video file rather than use the webcam due to technical issues on my end (couldnt get the camera to work)
video_capture = cv2.VideoCapture('videos/test_1.mp4')


# load sample pictures 
obama_image = fr.load_image_file('authorized_users/obama/obama_1.jpg')
trump_image = fr.load_image_file('authorized_users/trump/trump_1.jpg')
biden_image = fr.load_image_file('authorized_users/biden/biden_1.jpg')
will_image = fr.load_image_file('authorized_users/will/will_1.jpg')


# get face encoding
obama_face_encoding = fr.face_encodings(obama_image)[0]
trump_face_encoding = fr.face_encodings(trump_image)[0]
biden_face_encoding = fr.face_encodings(biden_image)[0]
will_face_encoding = fr.face_encodings(will_image)[0]

# create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    trump_face_encoding,
    biden_face_encoding,
    will_face_encoding
]

known_face_names = [
    "Barack Obama",
    "Donald Trump",
    "Joe Biden",
    "Will"
]

authorized_users = [
    "Barack Obama",
    "Donald Trump",
    "Joe Biden",
    "Will"
]

unauthorized_users = [
    "Unknown"
]

# initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_small_frame)

    if face_locations:
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # see if the faces match any known/authorized faces
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)    
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name in authorized_users:
                    print("Authorized User: ", name)
                    # this is where you would allow the user to enter their credentials to unlock whatever you are securing

                else:
                    print("Unauthorized User: ", name)
                    # this is where you would deny access to the user since they are not authorized to use the device

            face_names.append(name)

        # switch process from off to save processing power after every other frame
        process_this_frame = not process_this_frame
        
    else:
        print("No faces detected in the frame")
        continue  # Skip the rest of the loop if no faces are detected

    # display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # scale back up the face location since the frame was scaled down earlier
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # draw a box around the face
        # if authorized user (green), else unauthorized user (red)
        if name in authorized_users:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # draw a label with the name below the face
        # users name depends on authorization status (green = authorized, red = unauthorized)
        if name in authorized_users:
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # display the resulting image on screen
    cv2.imshow('Video', frame)
    
    # press 'esc' to quit
    if cv2.waitKey(27):
        break

# release handle to the webcam
# always do this
video_capture.release()
cv2.destroyAllWindows()

        


        

        


