import os
import cv2
import numpy as np
def train():
	print("training part initiated !")

	recog = cv2.face.LBPHFaceRecognizer_create()

	dataset = 'persons'

	paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]

	faces = []
	ids = []
	labels = []
	# for path in paths:
	# 	labels.append(path.split('/')[-1].split('-')[0])

	# 	ids.append(int(path.split('/')[-1].split('-')[2].split('.')[0]))

	# 	faces.append(cv2.imread(path, 0))
  
	for path in paths:
		# Assuming the filename format is "name-id-count-etc.jpg"
		parts = path.split(os.sep)[-1].split('_')  # os.sep is platform-independent
		label = parts[1]  # ID is now in the second position
		ids.append(int(label))
		faces.append(cv2.imread(path, 0))


	recog.train(faces, np.array(ids))

	recog.save('model.yml')

	return

train()