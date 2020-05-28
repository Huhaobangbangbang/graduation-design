import scipy.misc
face = scipy.misc.face()
face.shape(768, 1024, 3)
face.max()
face.dtype


import matplotlib.pyplot as plt

plt.gray()
plt.imshow(face)
plt.show()
