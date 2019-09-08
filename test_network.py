# USAGE
#-----------------------------------------------------
# python test_network.py --model Model.model --image test_samples/imagename.jpg

#------------------------------------------------------


from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()

image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model(args["model"])

(not_cat, cat) = model.predict(image)[0]

label = "cat" if cat > not_cat else "not_cat"
proba = cat if cat > not_cat else not_cat
label = "{}: {:.2f}%".format(label, proba * 100)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
cv2.imshow("Output", output)
cv2.waitKey(0)
