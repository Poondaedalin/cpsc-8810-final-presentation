from PIL import Image
import glob
import re

for f in glob.glob("thumbnails/*.png"):
    old_img = Image.open(f)
    new_img = old_img.crop((0, 45, 480, 315))
    new_img.save(f"cropped_thumbnails/{re.findall(r"(\d+.*\.png)", f)[0]}")