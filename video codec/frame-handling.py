def frame_decision(frames):
    
    for index, frame in enumerate(frames, start=1):
        if (index - 1) % 6 == 0:
            handle_iframe(frame, index)
        else:
            handle_pframe(frame, index)

def handle_iframe(frame, index):
    print(f"Handling I-frame at index {index}: {frame}")

def handle_pframe(frame, index):
    print(f"Handling P-frame at index {index}: {frame}")