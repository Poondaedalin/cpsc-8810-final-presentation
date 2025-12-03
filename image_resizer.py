from PIL import Image
import glob
import re

for file in glob.glob("thumbnails/*.png"):
    old_img = Image.open(file)
    new_img = old_img.resize((64,64), Image.LANCZOS)
    new_img.save(f"resized_thumbnails/{re.findall(r"(\d+.*\.png)", file)[0]}")