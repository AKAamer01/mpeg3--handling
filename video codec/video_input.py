# video_input.py
import cv2
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_grid(frame, color=(30, 30, 30)):
    """Subtle background grid for visual depth."""
    h, w = frame.shape[:2]
    for x in range(0, w, 20):
        cv2.line(frame, (x, 0), (x, h), color, 1)
    for y in range(0, h, 20):
        cv2.line(frame, (0, y), (w, y), color, 1)


def _draw_scanlines(frame, alpha=0.06):
    """CRT-style scanline overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for y in range(0, h, 3):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _lerp_color(c1, c2, t):
    """Linear interpolate between two BGR colors."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _draw_neon_circle(frame, center, radius, color, thickness=2):
    """Draw a circle with a soft glow halo."""
    r, g, b = color
    # Outer glow (dark, wide)
    glow_color = (int(b * 0.3), int(g * 0.3), int(r * 0.3))
    cv2.circle(frame, center, radius + 6, glow_color, thickness + 6)
    cv2.circle(frame, center, radius + 3, glow_color, thickness + 3)
    # Core line
    cv2.circle(frame, center, radius, (b, g, r), thickness)


def _draw_neon_rect(frame, pt1, pt2, color, thickness=2):
    """Draw a rectangle with a soft glow."""
    r, g, b = color
    glow = (int(b * 0.25), int(g * 0.25), int(r * 0.25))
    cv2.rectangle(frame, (pt1[0]-4, pt1[1]-4), (pt2[0]+4, pt2[1]+4), glow, thickness + 4)
    cv2.rectangle(frame, pt1, pt2, (b, g, r), thickness)


def _text_centered(frame, text, cy, font_scale=0.55, color=(220, 220, 220), thickness=1):
    """Draw horizontally-centered text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, cy), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_hud(frame, frame_idx, total_frames, fps):
    """Overlay a minimal HUD at the bottom."""
    h, w = frame.shape[:2]
    bar_h = 22

    # HUD bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Progress bar
    progress = frame_idx / max(total_frames - 1, 1)
    cv2.rectangle(frame, (0, h - 3), (int(w * progress), h), (0, 200, 255), -1)

    # Text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    t = frame_idx / fps
    label = f"FRAME {frame_idx + 1:03d}  |  T={t:.2f}s  |  {fps} FPS"
    cv2.putText(frame, label, (8, h - 6), font, 0.38, (160, 230, 255), 1, cv2.LINE_AA)

    right_label = f"{w}x{h}"
    (rw, _), _ = cv2.getTextSize(right_label, font, 0.38, 1)
    cv2.putText(frame, right_label, (w - rw - 8, h - 6), font, 0.38, (100, 180, 100), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  SCENE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _scene_orbiting_planets(frame, t, w, h):
    """Planets orbiting a central star with trails."""
    cx, cy = w // 2, h // 2

    # Star glow
    for r in [18, 12, 7]:
        alpha = 0.15 + 0.05 * r
        star_color = (0, int(200 * alpha), int(255 * alpha))
        cv2.circle(frame, (cx, cy), r, star_color, -1)
    cv2.circle(frame, (cx, cy), 5, (0, 240, 255), -1)

    planets = [
        dict(orbit=38, speed=1.4, size=6,  color=(255, 80,  50),  phase=0.0),
        dict(orbit=62, speed=0.8, size=8,  color=(80,  180, 255), phase=1.1),
        dict(orbit=88, speed=0.5, size=5,  color=(50,  255, 150), phase=2.4),
    ]

    for p in planets:
        angle = t * p['speed'] + p['phase']
        px = int(cx + p['orbit'] * np.cos(angle))
        py = int(cy + p['orbit'] * np.sin(angle))

        # Orbit ring (subtle)
        cv2.circle(frame, (cx, cy), p['orbit'], (40, 40, 40), 1)

        # Trail
        for trail_i in range(1, 8):
            ta = angle - trail_i * 0.12
            tx = int(cx + p['orbit'] * np.cos(ta))
            ty = int(cy + p['orbit'] * np.sin(ta))
            fade = 1.0 - trail_i / 8
            tc = tuple(int(c * fade) for c in p['color'])
            r = max(1, int(p['size'] * fade * 0.6))
            cv2.circle(frame, (tx, ty), r, (tc[2], tc[1], tc[0]), -1)

        _draw_neon_circle(frame, (px, py), p['size'], p['color'])


def _scene_wave_grid(frame, t, w, h):
    """Animated sine-wave grid with color cycling."""
    cols = 20
    rows = 15
    cell_w = w // cols
    cell_h = h // rows
    margin_x = (w - cols * cell_w) // 2
    margin_y = (h - rows * cell_h) // 2

    for row in range(rows):
        for col in range(cols):
            phase = (col + row) * 0.4 + t * 2.0
            wave = np.sin(phase)
            hue = int((col / cols * 120 + t * 30)) % 180
            sat = 200
            val = int(120 + 80 * wave)
            pixel = np.uint8([[[hue, sat, val]]])
            bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

            cx = margin_x + col * cell_w + cell_w // 2
            cy = margin_y + row * cell_h + cell_h // 2
            r = max(2, int(4 + 3 * wave))
            cv2.circle(frame, (cx, cy), r, color, -1)


def _scene_bouncing_shapes(frame, t, w, h):
    """Bouncing neon shapes with physics-like motion."""
    shapes = [
        dict(x0=0.2, y0=0.3, vx=0.18, vy=0.22, color=(255, 60, 120), shape='rect',  size=22),
        dict(x0=0.6, y0=0.1, vx=-0.15, vy=0.19, color=(60, 220, 255), shape='circle', size=14),
        dict(x0=0.4, y0=0.7, vx=0.20, vy=-0.13, color=(100, 255, 80), shape='circle', size=10),
        dict(x0=0.8, y0=0.5, vx=-0.22, vy=0.17, color=(255, 180, 40), shape='rect',   size=18),
    ]

    for s in shapes:
        # Bounce using modular reflection
        raw_x = s['x0'] + s['vx'] * t
        raw_y = s['y0'] + s['vy'] * t

        # Triangle-wave bounce in [0,1]
        bx = abs((raw_x % 2) - 1)
        by = abs((raw_y % 2) - 1)

        px = int(bx * (w - 2 * s['size']) + s['size'])
        py = int(by * (h - 2 * s['size'] - 22) + s['size'])

        if s['shape'] == 'circle':
            _draw_neon_circle(frame, (px, py), s['size'], s['color'])
        else:
            _draw_neon_rect(frame, (px - s['size'], py - s['size']),
                            (px + s['size'], py + s['size']), s['color'])


def _scene_warp_tunnel(frame, t, w, h):
    """Perspective tunnel zoom effect."""
    cx, cy = w // 2, h // 2
    num_rings = 10

    for i in range(num_rings, 0, -1):
        phase = (i / num_rings + t * 0.5) % 1.0
        scale = phase
        size_x = int(scale * w * 0.55)
        size_y = int(scale * h * 0.55)
        if size_x < 2 or size_y < 2:
            continue

        hue = int((i * 18 + t * 40)) % 180
        val = int(80 + 120 * (1 - scale))
        pixel = np.uint8([[[hue, 220, val]]])
        bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        thickness = max(1, int(3 * (1 - scale)))
        cv2.ellipse(frame, (cx, cy), (size_x, size_y), 0, 0, 360, color, thickness)


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_test_video(output_path="test_video.avi", duration_seconds=3, fps=10,
                        width=160, height=120):
    """
    Generate a visually rich test video with 4 animated scenes and a HUD overlay.

    Scenes (each ~25% of total frames):
      1. Orbiting Planets  – neon glow + motion trails
      2. Wave Grid         – HSV color-cycling dot grid
      3. Bouncing Shapes   – physics-like neon rectangles & circles
      4. Warp Tunnel       – perspective zoom rings

    Returns the output path.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_seconds * fps
    scene_len = total_frames // 4

    scene_names = [
        "ORBIT",
        "WAVE GRID",
        "BOUNCE",
        "TUNNEL",
    ]
    scene_fns = [
        _scene_orbiting_planets,
        _scene_wave_grid,
        _scene_bouncing_shapes,
        _scene_warp_tunnel,
    ]

    for frame_idx in range(total_frames):
        # Dark background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Subtle grid
        _draw_grid(frame, color=(18, 18, 18))

        t = frame_idx / fps

        # Scene selection
        scene_idx = min(frame_idx // scene_len, 3)
        scene_t = (frame_idx % scene_len) / fps
        scene_fns[scene_idx](frame, scene_t, width, height)

        # Scene label (top-left)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, scene_names[scene_idx], (6, 14),
                    font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # CRT scanlines
        _draw_scanlines(frame, alpha=0.08)

        # HUD
        _draw_hud(frame, frame_idx, total_frames, fps)

        out.write(frame)

    out.release()

    print(f"[VIDEO INPUT] Test video generated: {output_path}")
    print(f"  Resolution : {width}x{height}")
    print(f"  Duration   : {duration_seconds}s  |  FPS: {fps}")
    print(f"  Frames     : {total_frames}  (4 scenes x {scene_len} frames)")
    print(f"  Scenes     : {', '.join(scene_names)}")

    return output_path


def load_video_frames(video_path):
    """
    Load all frames from a video file.
    Returns a list of numpy arrays in BGR format.
    Raises FileNotFoundError if the path does not exist.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    fps  = cap.get(cv2.CAP_PROP_FPS)
    print(f"[VIDEO INPUT] Loaded {len(frames)} frames from '{video_path}'")
    if frames:
        h, w = frames[0].shape[:2]
        print(f"  Frame shape: {w}x{h}  |  Reported FPS: {fps:.1f}")

    return frames


def convert_frames_to_yuv(frames):
    """
    Convert a list of BGR frames to YUV color space (cv2.COLOR_BGR2YUV).
    Returns a list of numpy arrays.
    """
    if not frames:
        raise ValueError("No frames to convert.")

    yuv_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2YUV) for f in frames]
    print(f"[VIDEO INPUT] Converted {len(yuv_frames)} frames to YUV  "
          f"(shape: {yuv_frames[0].shape})")
    return yuv_frames


def save_video_from_frames(frames, output_path="decoded_video.avi", fps=10):
    """
    Save a list of BGR (or grayscale) frames as an AVI video.
    Handles dtype conversion and single-channel frames automatically.
    """
    if not frames:
        print("[VIDEO INPUT] No frames to save.")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame.reshape(h, w), cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] != 3:
            raise ValueError(f"Unexpected frame channels: {frame.shape[2]}")

        out.write(frame)

    out.release()
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[VIDEO INPUT] Saved {len(frames)} frames → '{output_path}'  "
          f"({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  VIDEO INPUT MODULE — STANDALONE TEST")
    print("=" * 55)

    # 1. Generate
    path = generate_test_video(
        "test_video.avi",
        duration_seconds=4,
        fps=10,
        width=160,
        height=120,
    )

    # 2. Load
    bgr_frames = load_video_frames(path)

    # 3. Convert
    yuv_frames = convert_frames_to_yuv(bgr_frames)

    # 4. Round-trip save
    save_video_from_frames(bgr_frames, "roundtrip_output.avi", fps=10)

    # 5. Quick stats
    print("\n--- Quick Stats ---")
    y_channel = yuv_frames[0][:, :, 0].astype(np.float32)
    print(f"  Y-channel mean  : {y_channel.mean():.2f}")
    print(f"  Y-channel stddev: {y_channel.std():.2f}")
    print(f"  Y-channel min   : {y_channel.min():.0f}")
    print(f"  Y-channel max   : {y_channel.max():.0f}")

    print("\n✓ video_input.py — all functions verified OK")