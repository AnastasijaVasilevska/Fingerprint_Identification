import multiprocessing as mp
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
start_time = time.time()

def getImagePaths(directory):
    imagePaths = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            imagePaths.append(os.path.join(directory, filename))

    return imagePaths


def getImages(imagePaths, downscale=False):
    images = []
    for directory in imagePaths:
        image = cv2.imread(directory)
        if downscale:
            h, w, c = image.shape
            if w >= 2500:
                image = cv2.pyrDown(image)
            if w >= 5000:
                image = cv2.pyrDown(image)
            if w >= 10000:
                image = cv2.pyrDown(image)
        images.append((directory, image))

    return images


class Sample:
    def __init__(self, path, image, keypoints, descriptors):
        self.path = path
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors

def main(queryImages, databaseImages):
    querySamples = []
    databaseSamples = []

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()

    print("Generating keypoints...")
    for image in queryImages:
        keypoints, descriptors = sift.detectAndCompute(image[1], None)
        querySamples.append(Sample(image[0], image[1], keypoints, descriptors))

    for image in databaseImages:
        keypoints, descriptors = sift.detectAndCompute(image[1], None)
        databaseSamples.append(Sample(image[0], image[1], keypoints, descriptors))

    print("Detecting best match...")
    results = []
    for i, querySample in enumerate(querySamples):
        perSampleResults = []
        for j, databaseSample in enumerate(databaseSamples):
            matches = flann.knnMatch(querySample.descriptors, databaseSample.descriptors, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) > 0:
                src_pts = np.float32([querySample.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([databaseSample.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                #mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)[1]
                perSampleResults.append((i, j, good))

        perSampleResults.sort(key=lambda x: len(x[2]), reverse=True)
        results.append(perSampleResults)

    if not os.path.exists("results"):
        os.mkdir("results")

    for i, result in enumerate(results):
        figure = plt.figure(querySamples[result[0][0]].path)

        img1 = querySamples[result[0][0]].image
        img2 = databaseSamples[result[0][1]].image

        result1 = cv2.drawMatches(img1, [], img2, [], [], None, (0, 255, 255), (0, 255, 255), None, 0)

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("Query Image + Best Match")

        plt.show()
        plt.close(figure)

if __name__ == "__main__":
    print("Loading images...")
    queryImagePaths = getImagePaths("query")
    databaseImagePaths = getImagePaths("database")
    queryImages = getImages(queryImagePaths)
    databaseImages = getImages(databaseImagePaths, True)

    procs = []

    proc = mp.Process(target=main, args=(queryImages, databaseImages[0:20]))
    procs.append(proc)
    proc.start()

    proc = mp.Process(target=main, args=(queryImages, databaseImages[20:40]))
    procs.append(proc)
    proc.start()

    proc = mp.Process(target=main, args=(queryImages, databaseImages[40:60]))
    procs.append(proc)
    proc.start()

    proc = mp.Process(target=main, args=(queryImages, databaseImages[60:80]))
    procs.append(proc)
    proc.start()

    for proc in procs:
        proc.join()

    #main(queryImages, databaseImages)
    print("My program took", time.time() - start_time, "to run")
