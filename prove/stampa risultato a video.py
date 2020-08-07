import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
img = cv.imread("../foto/add2.jpg")

# Text to be displayed
text = "800"

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

# Get the size of the text/result to be displayed
(result_width, result_height), baseline = cv.getTextSize(text, font, fontScale, thickness)

print("Text width: {}, height: {}".format(result_width, result_height))

# Using cv2.putText() method
img_with_text = cv.putText(img, text, org, font, fontScale, color, thickness, cv.LINE_AA)

# Displaying the image
plt.imshow(img_with_text)
plt.title("Prova")
plt.show()