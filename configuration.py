# Frame resizing height and width
frame_height, frame_width = 240, 320

# Number of frames for to extract features one time
frame_count = 16

# Number of segments to split a video
feature_per_bag = 32

# The weight path of abnormal situation detection model
classifier_model_weigts = './trained_models/weights-sultani.mat'

# The weight path of feature extraction model (C3D)
c3d_model_weights = './trained_models/c3d_sports1m.h5'

# Test Video Path
test_video_path = "C:/Users/dalab/Documents/Graduation work4/videos/was testing/014.mp4"