unzipped_object = zip(*samples)
# unzipped_sample = list(unzipped_object)

# w = np.ones(100000)*0.00001 #weights vector
# img = np.histogram2d(unzipped_sample[0], unzipped_sample[1], bins=[15,15], range=[[-7.5,7.5], [-7.5,7.5]], normed=None, weights=w, density=None)[0]
# img -= img.min()
# img /= img.max()
# img *= 255
# cv2.imshow('do not click x', img.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img[5:10,5:10])

