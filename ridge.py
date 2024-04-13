from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

#Ridge operators 
#https://scikit-image.org/docs/dev/auto_examples/edges/plot_ridge_filter.html#sphx-glr-auto-examples-edges-plot-ridge-filter-py
from skimage.filters import meijering, sato, frangi, hessian


img = io.imread("D:\PHOTOS AND PICTURES\Rocks_3.jpg")
img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#sharpened = unsharp_mask(image0, radius=1.0, amount=1.0)
meijering_img = meijering(img)
sato_img = sato(img)
frangi_img = frangi(img)
hessian_img = hessian(img)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(meijering_img, cmap='gray')
ax2.title.set_text('Meijering')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(sato_img, cmap='gray')
ax3.title.set_text('Sato')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(frangi_img, cmap='Reds_r')
ax4.title.set_text('Frangi')
plt.show()