# Face_Recognition_using_LBPH

This program will recognize the faces in the given video by detecting all the faces in the video, extracting the faces in it and training them using LBPH method. For face detection, Haar Cascade method is used. The LBPH method is used for face recognition.

To use the program, 
1. The faces to be trained are to be saved in the directories: 0, 1, 2 and so on. 
2. Run the train_model.py and give appropriate paths to the training images. This will generate a yml file of trained data.
3. Then, run the load_model_video.py and give path to the the yml file. Then, make sure the input video resolution and FPS is given correctly in the code. All the faces will be recognised.


