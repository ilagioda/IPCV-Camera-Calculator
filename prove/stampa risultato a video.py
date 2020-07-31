import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
img = cv.imread("../foto/add2.jpg")

# font
font = cv.FONT_HERSHEY_SIMPLEX

# coordinate (origine = angolo in alto a SX)
org = (3150, 1100)

# fontScale
fontScale = 14

# Black color
color = (0, 0, 0)

# Line thickness of 6 px
thickness = 6

# Using cv2.putText() method
img_with_text = cv.putText(img, '154', org, font, fontScale, color, thickness, cv.LINE_AA)

# Displaying the image
plt.imshow(img_with_text)
plt.title("Prova")
plt.show()