import numpy as np
import re
import glob

viewcounts = []

for img in glob.glob("thumbnails/*.png"):
    try:
        viewcounts.append(int(re.findall(r"(\d+)-YNO", str(img))[0]))
    except Exception as e:
        pass
    

viewcounts.sort(reverse=True)
print(viewcounts)

# viewcounts = viewcounts[:-5]

print(f"mean = {np.mean(viewcounts)}, median = {np.median(viewcounts)}, stdev = {np.std(viewcounts)}")