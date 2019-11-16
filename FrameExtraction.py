import numpy as np
import cv2
import os
import argparse
from os import listdir
import sys

parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--video_path', required=True, help='Path of video to extract frames')
parser.add_argument('--output_path', required=True, help='output folder for frames')
opt = parser.parse_args()
print(opt)

video_path = opt.video_path

output_path = opt.output_path

counter = 0

currentframe = 0
cap = cv2.VideoCapture(video_path)

while(True):
        
    ret, frame = cap.read()

    if np.shape(frame) == ():
        break

    if np.sum(frame) > 25000000:
        currentframe +=1
        frame_name = os.path.join(output_path,'frame'+ str(counter).zfill(5) + '.jpg')
   
        if (currentframe > 20000):
            if (currentframe % 1 == 0):  
                print('Creating...'+ frame_name)
                cv2.imwrite(frame_name, frame)
                counter += 1
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()

cv2.destroyAllWindows()
    
