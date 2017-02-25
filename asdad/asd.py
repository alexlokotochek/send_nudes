import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import dlib
from skimage import io

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
my_head_pose_estimator.load_yaw_variables("tensorflow/head_pose/yaw/cnn_cccdd_30k")
my_head_pose_estimator.load_pitch_variables("tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")

for i in range(1,3):
    file_name = str(i) + ".jpg"
    print("Processing image ..... " + file_name)
    image = cv2.imread(file_name) #Read the image with OpenCV
    yaw = my_head_pose_estimator.return_yaw(image)
    print("Estimated yaw ..... " + str(yaw[0,0,0]))
    print("")
    pitch = my_head_pose_estimator.return_pitch(image)
    print("Estimated pitch ..... " + str(pitch[0,0,0]))