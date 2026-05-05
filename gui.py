import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import os
import time
import numpy as np
import cv2
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODEC_DIR = os.path.join(BASE_DIR, "video codec")
sys.path.insert(0, CODEC_DIR)
import frame_handling
from video_input import generate_test_video, load_video_frames, convert_frames_to_yuv

class VideoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Codec Simulator")
        self.geometry("800x650")
        self.configure(bg="#1e1e1e")
        self.video_frames = []
        self._stop_playback = False
        self.setup_ui()

    def setup_ui(self):
        tk.Label(self, text="MPEG-3 Video Codec Simulator", font=("Arial", 16, "bold"), bg="#1e1e1e", fg="#00ff00").pack(pady=10)
        
        btn_frame = tk.Frame(self, bg="#1e1e1e")
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Generate Frames", command=self.on_generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Play Video", command=self.on_play).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Run Codec", command=self.on_compress).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.on_stop).pack(side=tk.LEFT, padx=5)
        
        self.canvas = tk.Canvas(self, width=480, height=320, bg="#000000", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        self.log_area = scrolledtext.ScrolledText(self, width=90, height=10, bg="#2d2d2d", fg="#00ff00", font=("Consolas", 10))
        self.log_area.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.update()

    def show_frame(self, frame, is_yuv=True):
        """Display a frame on the canvas. Accepts YUV or BGR."""
        if is_yuv:
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        else:
            bgr = frame
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((480, 320), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas._photo = photo

    def on_generate(self):
        threading.Thread(target=self._generate_worker, daemon=True).start()

    def on_play(self):
        self._stop_playback = False
        threading.Thread(target=self._play_worker, daemon=True).start()

    def on_compress(self):
        threading.Thread(target=self._compress_worker, daemon=True).start()

    def on_stop(self):
        self._stop_playback = True

    def _generate_worker(self):
        self.log("Generating test video (4 animated scenes)...")
        try:
            video_path = generate_test_video(
                output_path=os.path.join(BASE_DIR, "test_video.avi"),
                duration_seconds=4,
                fps=10,
                width=320,
                height=240,
            )
            self.log(f"Video saved to: {video_path}")

            bgr_frames = load_video_frames(video_path)
            self.log(f"Loaded {len(bgr_frames)} BGR frames.")

            yuv_frames = convert_frames_to_yuv(bgr_frames)
            self.log(f"Converted to YUV.")

            self.video_frames = yuv_frames
            self._bgr_frames   = bgr_frames   # keep for preview

            # Show a preview frame
            if bgr_frames:
                self.show_frame(bgr_frames[len(bgr_frames) // 2], is_yuv=False)

            self.log(f"{len(yuv_frames)} frames ready.")
        except Exception as e:
            self.log(f"ERROR during generation: {e}")

    def _play_worker(self):
        self.log("Playing video...")
        bgr_frames = getattr(self, '_bgr_frames', None)
        frames_to_play = bgr_frames if bgr_frames else self.video_frames
        is_yuv = bgr_frames is None
        delay = 1.0 / 10
        for frame in frames_to_play:
            if self._stop_playback:
                break
            self.show_frame(frame, is_yuv=is_yuv)
            time.sleep(delay)
        self.log("Playback finished.")

    def _compress_worker(self):
        self.log("Compressing video...")
        gop = 6
        frame_handling.REFERENCE_BUFFER.clear()
        for index, frame in enumerate(self.video_frames, start=1):
            if self._stop_playback:
                break
            if (index - 1) % gop == 0:
                compressed = frame_handling.compress_iframe(frame, index)
                frame_handling.REFERENCE_BUFFER.clear()
                decoded = frame_handling.decompress_iframe(compressed)
                frame_handling.REFERENCE_BUFFER.append(decoded)
                self.log(f"[I] frame {index}")
                self.show_frame(decoded)
            else:
                compressed = frame_handling.compress_pframe(frame, index)
                if compressed is None:
                    self.log(f"[P] frame {index} skipped")
                else:
                    decoded = frame_handling.decompress_pframe(compressed)
                    frame_handling.REFERENCE_BUFFER.append(decoded)
                    self.log(f"[P] frame {index}")
                    self.show_frame(decoded)
        self.log("Compression finished.")

if __name__ == "__main__":
    app = VideoApp()
    app.mainloop()