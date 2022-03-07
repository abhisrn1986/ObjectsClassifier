import os

from image_importer import get_image_data

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':

    base_path = f'{dir_path}/data/'
    classes = ["Forks", "Books", "Mugs", "Phones"]

    X, y = get_image_data(base_path, classes)
