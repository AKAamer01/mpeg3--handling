import cv2
import numpy as np
import frame_handling

width, height = 320, 240
fps = 30 
seconds = 5 
num_frames = fps * seconds 
frequency = 0.05 

def generate_video_input(): 
    frames_yuv = []
    
    print(f"Generating {num_frames} frames...")
    
    for i in range(num_frames):
        x = np.arange(width)
        line = 127.5 * (1 + np.cos(2 * np.pi * frequency * x + i * 0.2))  
        frame_gray = np.tile(line, (height, 1)).astype(np.uint8)
        
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        yuv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
        frames_yuv.append(yuv_frame)
        
    return frames_yuv

def verify_output(yuv_data):
    print("Playing video... Press 'q' to stop.")
    for frame in yuv_data:
        y_channel = frame[:, :, 0]
        
        cv2.imshow('Y-Channel (Luminance)', y_channel)
        
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

video_data = generate_video_input()
verify_output(video_data) 
frame_handling.frame_decision(video_data)