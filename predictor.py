import os
import time
import numpy as np
from c3d import *
from classifier import *
from utils.video_util import *
from utils.array_util import *
import configuration as cfg

def predict():
    video_path = cfg.test_video_path

    if not os.path.exists(video_path):
        exit("Video is not found")
    start_time = time.time()
    # Extract video clips for each 16 frames.
    video_clips, num_frames = get_video_clips(video_path)

    print(video_path, video_clips.shape, num_frames)

    # Build feature extraction (C3D) model.
    model_feature = c3d_feature_extractor()
    # Build feature classifier model.
    model_classifier = build_classifier_model()

    # Extract features
    
    features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        
        if len(clip) < cfg.frame_count:
            continue

        clip = preprocess_input(clip)
        feature = model_feature.predict(clip).squeeze()
        features.append(feature)

    features = np.array(features)

    # Split the features into Bags
    features_bag = interpolate(features, cfg.feature_per_bag)

    # Predict the features
    predictions = model_classifier.predict(features_bag)
    predictions = np.array(predictions).squeeze()

    # Set predictions into frames.
    predictions = extrapolate(predictions, num_frames)
    processed_sec = time.time() - start_time
    print("processed seconds", int(processed_sec))
    print(predictions.shape, len(predictions))
    for p in predictions:
        print(p)

if __name__ == "__main__":
    predict()