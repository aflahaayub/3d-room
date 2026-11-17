"""
Turn ONE photo of a room into a 3D point cloud.

Goal: give every pixel a 3D position (X, Y, Z) in metres, plus its colour,
and save all of them as a .ply file that the viewer can show.

The steps, in order (each step only uses things from the steps above it):
    STEP 1  Load the photo and make it smaller
    STEP 2  Ask Depth Anything for a depth number at every pixel
    STEP 3  Turn depth numbers into metres forward (Z), using one real measurement
    STEP 4  Find the lens zoom (focal length) in pixels
    STEP 5  Find the centre of the photo
    STEP 6  Turn pixel positions into metres sideways (X) and up/down (Y)
    STEP 7  Save the point cloud (.ply) and a check picture

Run it from anywhere:  uv run python src/3d_room_ai/depth_explained.py
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline


# ---------------------------------------------------------------------------
# SETTINGS - every number used below, and where it comes from
# ---------------------------------------------------------------------------

# This file is in src/3d_room_ai/, so the project folder is two levels up.
PROJECT_FOLDER = Path(__file__).resolve().parents[2]
OUTPUT_FOLDER = Path(__file__).resolve().parent

PHOTO_PATH = PROJECT_FOLDER / "IKEA.jpeg"
POINT_CLOUD_PATH = OUTPUT_FOLDER / "room-ikea-explained.ply"
CHECK_PICTURE_PATH = OUTPUT_FOLDER / "reference_check.png"

# CHOSEN: shrink the photo 4x so my laptop can handle it.
# 4000 x 3000 photo -> 1000 x 750 = 750,000 pixels = 750,000 points.
SHRINK_FACTOR = 4

# The reference object: RÅSKOG trolley (bottom right of the photo).
# PLACEHOLDER: 1.0 m is a guess. Measure it at IKEA and replace it.
# Measure straight forward from the camera (the direction the phone pointed),
# not the diagonal distance.
REFERENCE_DISTANCE_METRES = 1.0

# CHOSEN by looking at the photo: a pixel on the RÅSKOG top basket rim.
# These are positions in the SMALL (1000 x 750) image.
# Check reference_check.png after running: the red dot must sit ON the trolley.
REFERENCE_PIXEL_COLUMN = 895   # left -> right, 0 to 999
REFERENCE_PIXEL_ROW = 530      # top -> bottom, 0 to 749

# REAL (a standard): a 35 mm film frame is 36 mm x 24 mm.
# Its diagonal: sqrt(36^2 + 24^2) = 43.27 mm (calculated).
FILM_FRAME_DIAGONAL_MM = math.sqrt(36**2 + 24**2)


# ---------------------------------------------------------------------------
# STEP 1 - Load the photo and make it smaller
# ---------------------------------------------------------------------------

original_photo = Image.open(PHOTO_PATH)

# Phones save photos sideways and store "rotate me" in the file.
# exif_transpose applies that rotation so the photo is the right way up.
photo = ImageOps.exif_transpose(original_photo)

image_width = photo.width // SHRINK_FACTOR    # 4000 // 4 = 1000
image_height = photo.height // SHRINK_FACTOR  # 3000 // 4 = 750
small_photo = photo.resize((image_width, image_height))

print(f"Small photo size: {image_width} x {image_height} pixels")


# ---------------------------------------------------------------------------
# STEP 2 - Ask Depth Anything for a depth number at every pixel
# ---------------------------------------------------------------------------

depth_model = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",  # the small 25M version
)
model_result = depth_model(small_photo)

# depth_numbers is a grid: one number per pixel, shape (750 rows, 1000 columns).
# BIGGER number = CLOSER to the camera. The numbers are NOT metres.
depth_numbers = model_result["predicted_depth"].squeeze().cpu().numpy()

print(f"Depth grid shape: {depth_numbers.shape}")
print(f"Depth numbers range: {depth_numbers.min():.3f} to {depth_numbers.max():.3f}")


# ---------------------------------------------------------------------------
# STEP 3 - Turn depth numbers into metres forward (Z)
# ---------------------------------------------------------------------------
# We know ONE real distance: the RÅSKOG trolley is REFERENCE_DISTANCE_METRES away.
# Read the model's number at the trolley's pixel:
reference_depth_number = depth_numbers[REFERENCE_PIXEL_ROW, REFERENCE_PIXEL_COLUMN]

# Bigger number = closer, so distance goes the OPPOSITE way to the number:
#   a pixel with HALF the reference's number is TWICE as far away.
#
#   Z = (reference number / this pixel's number) * reference distance
#
# Same formula as the Roboflow tutorial.
# ASSUMPTION: the model's numbers only have an unknown scale, no unknown shift.
# If there is a shift, one measurement is not enough (needs 2 or more).
#
# This line does all 750,000 pixels at once (numpy works on the whole grid).
Z_metres_forward = (reference_depth_number / depth_numbers) * REFERENCE_DISTANCE_METRES

print(f"Reference depth number at RASKOG: {reference_depth_number:.3f}")


# ---------------------------------------------------------------------------
# STEP 4 - Find the lens zoom (focal length) in pixels
# ---------------------------------------------------------------------------
# The phone stores "focal length as if this were a 35 mm film camera" inside the
# photo file (EXIF). Tag 0x8769 is the group of camera settings; tag 0xA405 is
# "FocalLengthIn35mmFilm".
camera_settings = original_photo.getexif().get_ifd(0x8769)
focal_length_35mm = camera_settings.get(0xA405)

# Convert mm -> pixels by comparing diagonals:
#   film frame diagonal  = 43.27 mm
#   our photo's diagonal = sqrt(1000^2 + 750^2) = 1250 pixels (small photo)
# So 1 mm on the film frame = 1250 / 43.27 pixels in our photo.
photo_diagonal_pixels = math.sqrt(image_width**2 + image_height**2)
focal_length_pixels = focal_length_35mm * photo_diagonal_pixels / FILM_FRAME_DIAGONAL_MM

print(f"Focal length: {focal_length_35mm} mm (35 mm equivalent) = {focal_length_pixels:.1f} pixels")


# ---------------------------------------------------------------------------
# STEP 5 - Find the centre of the photo
# ---------------------------------------------------------------------------
# The camera's straight-ahead line passes through the middle of the photo.
# A pixel at the centre is straight ahead: X = 0, Y = 0.
# (ASSUMPTION: real cameras can be slightly off-centre; fine for this project.)
centre_column = image_width / 2    # 1000 / 2 = 500
centre_row = image_height / 2      # 750 / 2 = 375


# ---------------------------------------------------------------------------
# STEP 6 - Turn pixel positions into metres sideways (X) and up/down (Y)
# ---------------------------------------------------------------------------
# Every pixel's column number and row number, as two grids of shape (750, 1000).
pixel_columns, pixel_rows = np.meshgrid(np.arange(image_width), np.arange(image_height))

print(f"Pixel Columns: {pixel_columns} ")
print(f"Pixel Rows: {pixel_rows} ")

# Similar triangles (pinhole camera):
#   metres sideways / metres forward = pixels from centre / focal length in pixels
# so:
#   X = pixels from centre * Z / focal length
X_metres_sideways = (pixel_columns - centre_column) * Z_metres_forward / focal_length_pixels
Y_metres_down = (pixel_rows - centre_row) * Z_metres_forward / focal_length_pixels
# NOTE: rows count DOWNWARD in an image, so positive Y means "down".
# The viewer (index.html) flips this with rotation.x = Math.PI.


# ---------------------------------------------------------------------------
# STEP 7 - Save the point cloud (.ply) and a check picture
# ---------------------------------------------------------------------------
# Flatten the grids into long lists: one entry per pixel, 750,000 entries.
all_X = X_metres_sideways.ravel()
all_Y = Y_metres_down.ravel()
all_Z = Z_metres_forward.ravel()
all_colours = np.array(small_photo.convert("RGB")).reshape(-1, 3)  # red, green, blue
number_of_points = len(all_X)

with open(POINT_CLOUD_PATH, "w") as file:
    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write(f"element vertex {number_of_points}\n")
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property uchar red\n")
    file.write("property uchar green\n")
    file.write("property uchar blue\n")
    file.write("end_header\n")
    for x, y, z, (red, green, blue) in zip(all_X, all_Y, all_Z, all_colours):
        file.write(f"{x:.3f} {y:.3f} {z:.3f} {red} {green} {blue}\n")

print(f"Saved {number_of_points} points to {POINT_CLOUD_PATH}")

# Check picture: a red dot where we read the reference depth.
# Open it and make sure the dot is ON the RÅSKOG trolley, not in a gap.
check_picture = small_photo.copy()
draw = ImageDraw.Draw(check_picture)
dot_radius = 8
draw.ellipse(
    [
        REFERENCE_PIXEL_COLUMN - dot_radius,
        REFERENCE_PIXEL_ROW - dot_radius,
        REFERENCE_PIXEL_COLUMN + dot_radius,
        REFERENCE_PIXEL_ROW + dot_radius,
    ],
    outline="red",
    width=3,
)
check_picture.save(CHECK_PICTURE_PATH)
print(f"Saved reference check picture to {CHECK_PICTURE_PATH}")
