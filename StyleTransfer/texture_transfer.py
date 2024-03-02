"""
纹理迁移
"""

import cv2
import numpy as np

input_image = cv2.imread('./Image_mini/style2.jpg')
texture_image = cv2.imread('./Image_mini/dancing.jpg')
texture_image = cv2.resize(texture_image, (input_image.shape[1], input_image.shape[0]))

input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
texture_image_gray = cv2.cvtColor(texture_image, cv2.COLOR_BGR2GRAY)

diff = texture_image_gray - input_image_gray

output_image = np.zeros_like(input_image)
for channel in range(3):
    output_image[:, :, channel] = input_image[:, :, channel] + diff

cv2.imwrite('output_image.jpg', output_image)

