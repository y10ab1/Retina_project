import cv2
green_image = cv2.imread('../dataset/Training_Set/Training_Set/Training/1.png')

clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
b,g,r = cv2.split(green_image)
b = clahe.apply(b)
g = clahe.apply(g)
r = clahe.apply(r)
image = cv2.merge([b,g,r])
cv2.imwrite('original_CLAHE.png',image)

green_image[:,:,0] = 0
green_image[:,:,2] = 0
b,g,r = cv2.split(green_image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
b = clahe.apply(b)
g = clahe.apply(g)
r = clahe.apply(r)
image = cv2.merge([b,g,r])

cv2.imwrite('green_CLAHE.png',image)
cv2.imwrite('green.png',green_image)

