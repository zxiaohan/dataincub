import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg

#read the image
imagePath = 'dogcat/train/cats/cat.0.jpg'

# Read the image
image = mpimg.imread(imagePath)
print(image.shape)
plt.imshow(image)
plt.show()
