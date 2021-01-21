import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axisartist.axislines import Subplot
from PIL import Image


class face_generator:
    def __init__(self):
        pass

    def generate_face(self, img):
        im = Image.open(r"./paint/static/img/test.png")  

        return im


if __name__ == "__main__":
    a = face_generator()
    a.generate_face('aa')