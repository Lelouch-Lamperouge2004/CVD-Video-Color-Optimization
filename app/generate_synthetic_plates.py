import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

OUTPUT_DIR = r"D:\CVD_GAN\app\plates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDTH = 512
HEIGHT = 512
DOT_COUNT = 3000
DOT_RADIUS_RANGE = (6, 14)

try:
    FONT = ImageFont.truetype("arial.ttf", 200)
except:
    FONT = ImageFont.load_default()


def random_color(base_color, variation=35):
    return tuple(
        max(0, min(255, base_color[i] + random.randint(-variation, variation)))
        for i in range(3)
    )


def generate_plate(number_text, axis_type, filename):
    img = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Slightly vary base colors per plate
    if axis_type == "red-green":
        bg_base = (random.randint(80, 130), random.randint(150, 200), random.randint(80, 130))
        fg_base = (random.randint(150, 200), random.randint(80, 130), random.randint(80, 130))
    elif axis_type == "blue-yellow":
        bg_base = (random.randint(160, 210), random.randint(160, 210), random.randint(50, 90))
        fg_base = (random.randint(50, 90), random.randint(50, 90), random.randint(160, 210))
    else:
        bg_base = (150, 150, 150)
        fg_base = (200, 200, 200)

    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    mask_draw = ImageDraw.Draw(mask)

    bbox = mask_draw.textbbox((0, 0), number_text, font=FONT)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    mask_draw.text(
        ((WIDTH - text_width) / 2, (HEIGHT - text_height) / 2),
        number_text,
        fill=255,
        font=FONT
    )

    mask_np = np.array(mask)

    for _ in range(DOT_COUNT):
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
        r = random.randint(*DOT_RADIUS_RANGE)

        if mask_np[y, x] > 0:
            color = random_color(fg_base)
        else:
            color = random_color(bg_base)

        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)

    img.save(os.path.join(OUTPUT_DIR, filename))


def main():
    red_green_numbers = ["12", "8", "6", "57", "45", "73"]
    blue_yellow_numbers = ["29", "5", "3", "64", "91", "27"]

    for i, num in enumerate(red_green_numbers):
        generate_plate(num, "red-green", f"plate_rg_{i+1}.png")

    for i, num in enumerate(blue_yellow_numbers):
        generate_plate(num, "blue-yellow", f"plate_by_{i+1}.png")

    print("12 synthetic plates generated in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()