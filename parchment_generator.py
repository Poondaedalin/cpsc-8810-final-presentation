from PIL import Image
import glob
import re

path = "results/regression in generator/r=500/*.png"

canvas = Image.new("RGB", (640, 640))

file_glob = glob.glob(path)
file_glob.sort(key = lambda x: int(re.findall(r"(\d+)\.png", x)[0]))

for i, img in enumerate(file_glob):
    x = (i) % 10
    y = (i) // 10

    img_copy = Image.open(img)

    canvas.paste(img_copy,(x * 64, y * 64))

canvas.save("parchment.png")