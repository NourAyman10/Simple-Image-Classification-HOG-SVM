import os
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.metrics import accuracy_score

images_classes = ['accordian', 'dollar_bill', 'motorbike', 'Soccer_Ball']


def get_variables(images_directory):
    x = []
    y = []
    for image_class_index, image_class in enumerate(images_classes):
        for file in os.listdir(os.path.join(images_directory, image_class)):
            image = prepare_images(file, image_class, images_directory)
            x.append(get_hog_feature(image))
            y.append(image_class_index)
    return [x, y]


def prepare_images(file, image_class, images_directory):
    image_path = os.path.join(images_directory, image_class, file)
    image = imread(image_path)
    image = resize(image, (128, 64))
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    image.flatten()
    return image


def get_hog_feature(image):
    feature, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=True, multichannel=True, feature_vector=True)
    return feature


# get train and test variables
X_train, Y_train = get_variables("train")
X_test, Y_test = get_variables("test")

for kernel in ("linear", "poly", "rbf"):
    # train classifier
    svm_classifier = svm.SVC(kernel=kernel, gamma=2)
    svm_classifier.fit(X_train, Y_train)

    # test performance
    Y_prediction = svm_classifier.predict(X_test)
    print("Accuracy "+kernel+": " + str(accuracy_score(Y_test, Y_prediction) * 100), "%")
    print("******************************************************")
