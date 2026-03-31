import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import whisper
import subprocess
import os
import textwrap




def show_image(img_np):
    """
    Uses matplotlib to display an input np.ndarray image.

    img_np: np.ndarray (H, W, 3), RGB, uint8
    """
    plt.figure(figsize=(20, 8))
    plt.imshow(img_np)
    plt.axis("off")
    plt.show()




def render_text(
    text,
    font_path='../resources/fonts/zxx-noise.ttf',
    font_size=80,
    canvas_size=(900, 300),
    x_cursor=50,
    params=None
):
    # Normalize text input
    if isinstance(text, list):
        first = text[0]
        text = first if isinstance(first, str) else first["text"]

    if params is None:
        params = {}

    # Default params
    p = {
        "per_char_rot_std": 0.0,
        "kerning_std": 0.0,
        "baseline_amp": 0.0,
        "baseline_freq": 0.0,
        "y_jitter_std": 0.0,
        "text_color": (255, 255, 255),
        "background_color": (0, 0, 0),
        "text_background_color": None,
        "text_background_padding": 5
    }
    p.update(params)

    img = Image.new("RGB", canvas_size, p["background_color"])
    font = ImageFont.truetype(font_path, font_size)

    ascent, descent = font.getmetrics()
    baseline_y = canvas_size[1] // 2

    rot_freq = 1
    jitter_freq = 1
    kern_freq = 1

    for i, char in enumerate(text):

        temp_img = Image.new("L", (500, 500), 0)
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), char, font=font)

        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])

        margin = 20
        char_img = Image.new(
            "RGBA",
            (w + margin * 2, h + margin * 2),
            (0, 0, 0, 0)
        )
        char_draw = ImageDraw.Draw(char_img)

        # Draw background box behind character
        if p["text_background_color"] is not None:
            pad = p["text_background_padding"]
            char_draw.rectangle(
                [margin - pad, margin - pad,
                 margin + w + pad, margin + h + pad],
                fill=(*p["text_background_color"], 255)
            )

        char_draw.text(
            (margin - bbox[0], margin - bbox[1]),
            char,
            font=font,
            fill=p["text_color"]
        )

        # per-character rotation
        rot = p["per_char_rot_std"] * np.sin(rot_freq * i)
        char_img = char_img.rotate(rot, expand=True)

        # baseline curve
        baseline_offset = (
            p["baseline_amp"]
            * np.sin(p["baseline_freq"] * i)
        )

        # vertical jitter
        y_jitter = (
            p["y_jitter_std"]
            * np.sin(jitter_freq * i)
        )

        y_pos = int(
            baseline_y
            - ascent
            + baseline_offset
            + y_jitter
        )

        img.paste(char_img, (int(x_cursor), int(y_pos)), char_img)

        # kerning
        kerning = (
            p["kerning_std"]
            * np.sin(kern_freq * i)
        )

        x_cursor += max(1, w + 10 + kerning)

    return np.array(img, dtype=np.uint8)


def wrap_and_render(text, vid_width, subtitle_height, max_chars, line_spacing,
                    font_path, font_size, x_cursor, params):
    """Wrap text and render it into a subtitle strip."""
    lines = textwrap.wrap(text, width=max_chars) or [text]
    n = len(lines)
    line_height = max(1, (subtitle_height - line_spacing * (n - 1)) // n)
    bg = params.get("background_color", (0, 0, 0)) if params else (0, 0, 0)

    line_imgs = []
    for i, line in enumerate(lines):
        rendered = render_text(
            line,
            font_path=font_path,
            font_size=font_size,
            canvas_size=(vid_width, line_height),
            x_cursor=x_cursor,
            params=params
        )
        line_imgs.append(rendered)
        if line_spacing > 0 and i < len(lines) - 1:
            spacer = np.full((line_spacing, vid_width, 3), bg, dtype=np.uint8)
            line_imgs.append(spacer)

    stacked = np.vstack(line_imgs)
    current_h = stacked.shape[0]
    if current_h < subtitle_height:
        pad = np.full((subtitle_height - current_h, vid_width, 3), bg, dtype=np.uint8)
        stacked = np.vstack([stacked, pad])

    return stacked[:subtitle_height]


def _composite(frame_rgb, subtitle_rgb, y_position, subtitle_height, params):
    """Stamp a subtitle strip onto an RGB frame."""
    vid_height, vid_width = frame_rgb.shape[:2]
    bg = params.get("background_color", (0, 0, 0)) if params else (0, 0, 0)
    text_bg = params.get("text_background_color", None) if params else None

    subtitle_bgr = cv2.cvtColor(subtitle_rgb, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    y1 = max(0, y_position)
    y2 = min(vid_height, y_position + subtitle_height)
    strip_y1 = y1 - y_position
    strip_y2 = strip_y1 + (y2 - y1)

    if text_bg is not None:
        # Paste entire subtitle strip — background boxes and all
        frame_bgr[y1:y2] = subtitle_bgr[strip_y1:strip_y2]
    else:
        # Only stamp text pixels, video shows through elsewhere
        bg_bgr = np.array([bg[2], bg[1], bg[0]], dtype=np.uint8)
        mask = np.any(subtitle_bgr != bg_bgr, axis=2)
        region = frame_bgr[y1:y2]
        strip = subtitle_bgr[strip_y1:strip_y2]
        strip_mask = mask[strip_y1:strip_y2]
        region[strip_mask] = strip[strip_mask]
        frame_bgr[y1:y2] = region

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def render_subtitles(
    subtitles,
    output_path=None,
    video_path=None,
    frame=None,
    font_path='../resources/fonts/zxx-noise.ttf',
    font_size=80,
    x_cursor=50,
    y_position=None,
    subtitle_height=150,
    max_chars=40,
    line_spacing=10,
    text_background_color=(0, 0, 0),   # background color behind each character
    text_background_padding=5,          # padding around each character background
    params=None
):
    assert video_path is not None or frame is not None, \
        "Must provide either video_path or frame"

    # Inject background args into params so render_text picks them up
    if params is None:
        params = {}
    params["text_background_color"] = text_background_color
    params["text_background_padding"] = text_background_padding

    
    # Normalize subtitles input
    if isinstance(subtitles, str):
        subtitles = [{"text": subtitles}]
    elif isinstance(subtitles, list) and len(subtitles) > 0 and isinstance(subtitles[0], str):
        subtitles = [{"text": "\n".join(subtitles)}]

    # --- Single frame mode ---
    if frame is not None:
        vid_height, vid_width = frame.shape[:2]

        if y_position is None:
            y_position = vid_height - subtitle_height - 40

        text = subtitles[0]["text"]
        subtitle_rgb = wrap_and_render(
            text, vid_width, subtitle_height, max_chars,
            line_spacing, font_path, font_size, x_cursor, params
        )
        result = _composite(
            frame.copy(), subtitle_rgb, y_position, subtitle_height, params
        )

        if output_path:
            Image.fromarray(result).save(output_path)

        return result

    # --- Video mode ---
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if y_position is None:
        y_position = vid_height - subtitle_height - 40

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (vid_width, vid_height))

    subtitle_cache = {}

    def get_subtitle_for_time(t):
        for entry in subtitles:
            if entry['start'] <= t < entry['end']:
                return entry['text']
        return None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        text = get_subtitle_for_time(t)

        if text is not None:
            if text not in subtitle_cache:
                subtitle_cache[text] = wrap_and_render(
                    text, vid_width, subtitle_height, max_chars,
                    line_spacing, font_path, font_size, x_cursor, params
                )
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            composited = _composite(
                frame_rgb, subtitle_cache[text], y_position, subtitle_height, params
            )
            frame = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved to {output_path} ({frame_idx} frames at {fps:.2f} fps)")

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extract audio from video using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",  # mono, 16kHz (Whisper's preferred format)
        audio_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def transcribe_to_subtitles(video_path, model_size="base", mode="segment"):
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)

    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path,
        word_timestamps=(mode == "word")  # enables per-word timing
    )

    subtitles = []

    if mode == "word":
        for segment in result["segments"]:
            for word in segment["words"]:
                subtitles.append({
                    "text": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"]
                })

    elif mode == "segment":
        for segment in result["segments"]:
            subtitles.append({
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"]
            })

    os.remove(audio_path)
    return subtitles

def replace_audio(source_video, audio_video, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", source_video,
        "-i", audio_video,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-shortest",
        output_path
    ], check=True)

def get_frame_at_time(video_path, t):
    """
    Extract a single frame from a video at a specific timestamp.
    The parameters t is in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame at t={t}s")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)