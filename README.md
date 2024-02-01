# face_tracking_pypi


face tracking and user recognition from a video source for a fullsail university: Computer Vision class project


January 2024


used a openCV deep learning face detector model based on the caffe framework for the backbone for this project. I then trained it on a pre-defined set of facial images
for 'authorized users' that it would compare the faces in the video to, and if found would label authorized and the box would change to a green color, however, 
if the face was not found in the pre-defined list it would be named unauthorized and the box would change to a red color. I used face_recognition for the encoding of the 
known authorized faces for the model to find.