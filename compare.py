import glob

import cv2
import matplotlib.pyplot as plt

index = {}
images = {}
div = 64

def reduceColor():
    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]
    for x in range(0, h):
        for y in range(0, w):
            for z in range(0, c):
                image[x, y, z] = image[x, y, z] / div * div + div / 2


for imagePath in glob.glob("images/*.png"):
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    # reduceColor()
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, div, 0, div, 0, div])
    hist = cv2.normalize(hist, hist).flatten()
    index[filename] = hist

OPENCV_METHODS = (
    # ("Correlation", cv2.HISTCMP_CORREL),
    # ("Chi-Squared", cv2.HISTCMP_CHISQR),
    # ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA),
)

for (methodName, method) in OPENCV_METHODS:
    results = {}
    reverse = False

    if methodName in ("Correlation", "Intersection"):
        reverse = True

    for (k, hist) in index.items():
        d = cv2.compareHist(index["doge.png"], hist, method)
        results[k] = d

    results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)

    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(images["doge.png"])
    plt.axis("off")

    fig = plt.figure("Results: %s" % (methodName))
    fig.suptitle(methodName, fontsize=20)

    for (i, (v, k)) in enumerate(results):
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.set_title("%.2f" % v)
        plt.imshow(images[k])
        plt.axis("off")

plt.show()
