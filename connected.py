import cv2
import numpy as np
import matplotlib.pyplot as plt

N=50
img = np.random.random((N,N))
for i in range(N):
    for j in range(N):
        if img[i][j]>.6:
            img[i][j]=1
        else:
            img[i][j]=0

new_img = np.zeros_like(img)
for val in np.unique(img)[1:]:
    mask = np.uint8(img == val)
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    new_img[labels == largest_label] = val

print(new_img)
plt.matshow(new_img,cmap='Purples')
plt.axis('off')
plt.show()
