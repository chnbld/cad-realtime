import os
import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from c3d import *
from classifier import *
from utils.video_util import *
from utils.array_util import *
import configuration as cfg
import time
import threading

last_error = 0.0
num_frames = cfg.frame_count

processed_sec = 0.0
last_score = 0.0

frames = []

def extract_features(video_clips):
    video_clips = sliding_window(frames, params.frame_count, params.frame_count)
    clips = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < cfg.frame_count:
            continue
        
        clip = preprocess_input(clip)
        clips.append(clip)
        
        clips = np.vstack(clips)
        features = model_feature.predict(clips)

        #print("Feature Extract: %.2f" % (time.time() - start_time),  end=" ", flush=True)

        features_bag = interpolate(features, cfg.feature_per_bag)
        predictions = model_classifier.predict(features_bag)
        predictions = np.array(predictions).squeeze()
        predictions = extrapolate(predictions, num_frames)

        last_score = np.max(predictions)
        processed_sec = time.time() - start_time

        #print("Prediction: %.2f," % (processed_sec), end=" ", flush=True)

        #print("Max Score: %.2f" % (last_score))
        print("%.2f" % (last_score))

        #window_title = "Frames: %d, Processod sec: %.2f, Score: %.2f" % (num_frames, processed_sec, last_score)
        #cv2.setWindowTitle("Frame", window_title)

        return features

if __name__ == "__main__":
    model_feature = c3d_feature_extractor()
    model_classifier = build_classifier_model()

    cap = cv2.VideoCapture("C:/Users/dalab/Documents/Graduation work4/videos/was testing/107.mp4")#cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            
            window_title = "Frames: %d, Processod sec: %.2f, Score: %.2f" % (num_frames, processed_sec, last_score)
            cv2.imshow("Frame", frame)
            
    
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if len(frames) == num_frames:
                start_time = time.time()

                video_clips = sliding_window(frames, params.frame_count, params.frame_count)
                #print("Time (seconds): video_clips: %.2f" % (time.time() - start_time), end=" ", flush=True)

                # features = extract_features(video_clips)

                x = threading.Thread(target=extract_features, args=(video_clips,))
                x.start()
                
                frames = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by q")
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()        