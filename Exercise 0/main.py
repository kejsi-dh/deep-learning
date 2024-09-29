from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

def main():

    """ EXERCISE 2 """
    """
    checker = Checker(res = 400, tile_size = 50)
    checker.draw()
    checker.show()

    circle = Circle(res = 400, r = 100, pos = (200, 200))
    circle.draw()
    circle.show()

    spectrum = Spectrum(res = 400)
    spectrum.draw()
    spectrum.show()
    """
    #################################################

    """ EXERCISE 2 """

    file_path = '/Users/Kejsi/Documents/school/Msc. AI/Second Semester/Deep Learning/Exercises/exercise0_material/src_to_implement/data/exercise_data/'
    label_path = '/Users/Kejsi/Documents/school/Msc. AI/Second Semester/Deep Learning/Exercises/exercise0_material/src_to_implement/data/Labels.json'
    batch = 12
    image = [32, 32, 3]

    gen = ImageGenerator(file_path, label_path, batch, image,
                         rotation=False, mirroring=True, shuffle=True)

    gen.show()
    print("Current Epoch: ", gen.current_epoch())
    images, labels = gen.next()

if __name__ == "__main__":
    main()