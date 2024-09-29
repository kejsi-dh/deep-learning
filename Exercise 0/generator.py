import os.path
import json
from scipy.ndimage import rotate
import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle as rand_shuffle

# In this exercise task you will implement an image generator.
# Generator objects in python are defined as having a next function.
# This next function returns the next generated object.
# In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:

    # TODO: implement constructor
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False,
                 mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle_flag = shuffle
        self.current_idx = 0
        self.epoch = 0

        with open(label_path, 'r') as f: self.labels = json.load(f)

        # ensure the filenames end with ".npy"
        self.image_files = [f"{key}.npy" if not key.endswith('.npy')
                            else key for key in self.labels.keys()]
        self.num_images = len(self.image_files)

        if self.shuffle_flag: self._shuffle_data()

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        images = []
        labels = []

        for _ in range(self.batch_size):
            if self.current_idx >= self.num_images:
                self.epoch += 1
                self.current_idx = 0
                if self.shuffle_flag:
                    self._shuffle_data()

            img_file = self.image_files[self.current_idx] # current image
            img = self._load_image(img_file) # load
            img = self.augment(img) # augment
            img = np.resize(img, self.image_size) #resize

            images.append(img)
            labels.append(self.labels[img_file.replace(".npy", "")])

            self.current_idx += 1

        # convert to arrays
        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def _shuffle_data(self):
        combined = list(self.image_files)
        rand_shuffle(combined)
        self.image_files = combined

    def _load_image(self, filename):
        filepath = os.path.join(self.file_path, filename)
        if os.path.exists(filepath): return np.load(filepath)
        else: raise FileNotFoundError(f"File {filepath} not found.")

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        # mirroring
        if self.mirroring and randint(0, 1): img = np.fliplr(img)

        # rotation
        if self.rotation:
            rotations = [0, 90, 180, 270]
            angle = rotations[randint(0, 3)]
            if angle != 0: img = rotate(img, angle, reshape=False)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function

        class_dict = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog',
            6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }
        return class_dict.get(x, "Unknown")

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method

        images, labels = self.next()

        plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(1, self.batch_size, i + 1)
            plt.imshow(images[i])
            plt.title(self.class_name(labels[i]))
            plt.axis('off')

        plt.show()

