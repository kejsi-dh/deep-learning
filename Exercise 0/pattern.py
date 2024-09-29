import numpy as np
import matplotlib.pyplot as plt

# checkerboard
class Checker:
    def __init__(self, res, tile_size):
        if res % (2 * tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2 * tile_size without remainder.")

        self.res = res
        self.tile_size = tile_size
        self.output = None

    # create the checkerboard pattern as a numpy array
    def draw(self):
        num_tiles = self.res // self.tile_size
        self.output = np.zeros((self.res, self.res), dtype = int)

        for i in range(num_tiles):
            for j in range(num_tiles):
                if (i + j) % 2 == 0:
                    self.output[i * self.tile_size:(i + 1) * self.tile_size,
                    j * self.tile_size:(j + 1) * self.tile_size] = 0
                else:
                    self.output[i * self.tile_size:(i + 1) * self.tile_size,
                    j * self.tile_size:(j + 1) * self.tile_size] = 255

        return self.output.copy()

    # show the checkerboard pattern
    def show(self):
        if self.output is None:
            raise RuntimeError("The pattern has not been drawn yet. Call the draw() method first.")

        plt.imshow(self.output, cmap = 'gray')
        plt.title("Checkerboard")
        plt.show()

# binary circle
class Circle:
    def __init__(self, res, r, pos):
        self.res = res
        self.r = r
        self.pos = pos
        self.output = None

    # create a binary image of a circle as a numpy array
    def draw(self):
        self.output = np.zeros((self.res, self.res), dtype = int)

        for x in range(self.res):
            for y in range(self.res):
                dist = np.sqrt((x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2)
                if dist <= self.r: self.output[x, y] = 255

        return self.output.copy()

    # show the circle
    def show(self):
        if self.output is None:
            raise RuntimeError("The pattern has not been drawn yet. Call the draw() method first.")

        plt.imshow(self.output, cmap = 'gray')
        plt.title("Circle")
        plt.show()

# spectrum
class Spectrum:
    def __init__(self, res):
        self.res = res
        self.output = None

    # create the spectrum as a numpy array
    def draw(self):
        self.output = np.zeros((self.res, self.res, 3), dtype = float)

        for i in range(self.res):
            for j in range(self.res):
                red = 0.9 * (j / (self.res - 1))  # red colour: left to right
                green = 0.9 * (i / (self.res - 1))  # green colour: top to bottom
                blue = 1.5 * ((i / (self.res - 1)) * (j / (self.res - 1)))  # blue colour: diagonal

                self.output[i, j, 0] = red
                self.output[i, j, 1] = green
                self.output[i, j, 2] = blue

        return self.output.copy()

    # show the spectrum
    def show(self):
        if self.output is None:
            raise RuntimeError("The pattern has not been drawn yet. Call the draw() method first.")

        plt.imshow(self.output)
        plt.title("RGB Spectrum")
        plt.show()