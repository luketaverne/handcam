import numpy as np
import cv2
import scipy.misc
from scipy.ndimage.interpolation import shift

# Read the images
# foreground = cv2.imread("/local/home/luke/datasets/handcam/greenscreens/88-right2-image.png")
# background = cv2.imread("beach-drink.jpg")
# alpha = cv2.imread("/local/home/luke/datasets/handcam/greenscreens/88-right2-mask.png")
foreground = cv2.imread("1-right-image.png")
background = cv2.imread("beach-drink.jpg")
alpha = cv2.imread("1-right-mask.png")


hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV) #convert it to hsv

h, s, v = cv2.split(hsv)
lim = 255 - 100
v[ v> lim] =  255
v[v<=lim] += 100
final_hsv = cv2.merge((h, s, v))

foreground = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

foreground = shift(foreground,[0,-170,0])
alpha = shift(alpha,[0,-170,0])

# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)

# Normalize the alpha mask to keep intensity between 0 and 1
alpha = alpha.astype(float) / 255

# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)

# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)

# Add the masked foreground and background.
outImage = cv2.add(foreground, background)

# Save img
scipy.misc.imsave("beach-hand-bad.png", cv2.cvtColor(outImage.astype(np.uint8),cv2.COLOR_BGR2RGB), format="png")

# Display image
cv2.imshow("outImg", outImage / 255)
cv2.waitKey(34)

while True:
    pass