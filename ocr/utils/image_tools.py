import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont




def show_image(img_np):
    """
    Uses matplotlib to display an input np.ndarray image.

    img_np: np.ndarray (H, W, 3), RGB, uint8
    """
    plt.figure(figsize=(10, 4))
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
        "background_color": (0, 0, 0)
    }
    p.update(params)

    img = Image.new("RGB", canvas_size, p["background_color"])
    font = ImageFont.truetype(font_path, font_size)

    ascent, descent = font.getmetrics()
    baseline_y = canvas_size[1] // 2

    """
    Each transformation is made deterministic through the use of np.sin with
        parameterized standard deviation (centered around 0)
    """
    rot_freq = 1
    jitter_freq = 1
    kern_freq = 1

    for i, char in enumerate(text):

        # Measure bounding box
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