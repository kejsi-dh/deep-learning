# 1 Array Manipulation Warm-up

## 1.1 Checkerboard
The first pattern to implement is a checkerboard pattern in the class _Checker_ with adaptable tile size and resolution. You might want to start with a fixed tile size and adapt later on. For simplicity we assume that the resolution is divisible by the tile size without remainder.

Task:
* Implement the constructor. It receives two arguments: an integer resolution that de- fines the number of pixels in each dimension, and an integer tile size that defines the number of pixel an individual tile has in each dimension. Store the arguments as instance variables. Create an additional instance variable output that can store the pattern.
* Implement the method draw() which creates the checkerboard pattern as a numpy array. The tile in the top left corner should be black. In order to avoid truncated checkerboard patterns, make sure your code only allows values for resolution that are evenly dividable by 2· tile size. Store the pattern in the instance variable output and return a copy. Helpful functions for that can be found on the Deep Learning Cheatsheet provided.
* Implement the method show() which shows the checkerboard pattern with for example plt.imshow(). If you want to display a grayscale image you can use cmap = gray as a parameter for this function.

## 1.2 Circle
The second pattern to implement is a binary circle with a given radius at a specified position in the image.

Task:
* Implement the constructor. It receives three arguments: An integer resolution, an integer radius that describes the radius of the circle, and a tuple position that contains the x- and y-coordinate of the circle center in the image.
* Implement the method draw() which creates a binary image of a circle as a numpy array. Store the pattern in the instance variable output and return a copy.
* Implement the method show() which shows the circle with for example plt.imshow().

## 1.3 RGB Spectrum
The third pattern to implement is an RGB color spectrum.

Task:
* Implement the constructor. It receives one parameter: an integer resolution.
* Implement the method draw() which creates the spectrum as a numpy array. Remember that RGB images have 3 channels and that a spectrum consists of rising values across a specific dimension. For each color channel, the intensity minimum and maximum should be 0.0 and 1.0, respectively. Store the pattern in the instance variable output and return a copy.
* Implement the method show() which shows the RGB spectrum with for example plt.imshow().

# 2 Data Handling Warmup

## Image Generator
We will implement a class that is able to read in a set of images, their associated class labels (stored as a JSON file), and generate batches (subsets of the data) that can be used for training of a neural network.

Task:
* Implement the class ImageGenerator in the file “generator.py”.
* Provide a constructor receiving
  1. the path to the directory containing all images file path as a string
  2. the path to the JSON file label path containing the labels again as string
  3. an integer batch size defining the number of images in a batch.
  4. a list of integers defining the desired image size [height, width, channel]
  5. and optional bool flags rotation, mirroring, shuffle which default to False.
* The labels in the JSON file are stored as a dictionary, where the key represents the corresponding filename of the images as a string (e.g. the key ’15’ corresponds to the image 15.npy) and the value of each key stands for the respective class label encoded as integer. (0 = ’airplane’; 1 = ’automobile’; 2 = ’bird’; 3 = ’cat’; 4 = ’deer’; 5 = ’dog’; 6 = ’frog’; 7 = ’horse’; 8 = ’ship’; 9 = ’truck’ )
* Provide the method next(), which returns one batch of the provided dataset as a tuple (images, labels), where images represents a batch of images and labels an array with the corresponding labels, when called. Each image of your data set should be included only once in those batches until the end of one epoch. One epoch describes a run through the whole data set. A resizing option should be included within the next() method. Make sure all your batches have the same size. If the last batch is smaller than the others, complete that batch by reusing images from the beginning of your training data set.
* Implement the following functionalities for data manipulation and augmentation:
  1. shuffle: If the shuffle flag is True, the order of your data set (= order in which the images appear) is random (Not only the order inside one batch!).
  2. mirroring: If the mirroring flag is True, randomly mirror the images in the method next().
  3. rotation: If the rotation flag is True, randomly rotate the images by 90, 180 or 270◦ in the method next()
* Implement a method current epoch() which returns an integer of the current epoch. This number should be updated in the next() when we start to iterate through the data set (again) from the beginning.
* Implement a method class name(int label), which returns the class name that corre- sponds to the integer label in the argument int label.
* Implement a method show() which generates a batch using next() and plots it. Use class name() to obtain the titles for the image plots






