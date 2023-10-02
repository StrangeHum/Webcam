import numpy as np
from nNetw import NeuralNetwork


def getArrayFromFile(src):
    from PIL import Image

    srcImage = Image.open(src)
    grayImage = srcImage.convert("L")
    array = np.array(grayImage)

    return array


array = getArrayFromFile("./data/Glassmorphism Big Sur Messenger App Redesign.jpg")

neuro = NeuralNetwork([2, 1])

err = 100

while err > 0.001:
    value, err = neuro.correct([2, 1], [3])
    print(value[-1])

# neuro.save("./weight/w.neuro.json")
