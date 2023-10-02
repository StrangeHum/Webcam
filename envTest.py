from nNetw import NeuralNetwork

neuro = NeuralNetwork([2, 1])

print(neuro.out([2, 1])[-1])

err = 100

while err > 0.001:
    value, err = neuro.correct([2, 1], [3])
    print(value[-1])

print(neuro.out([2, 1])[-1])

neuro.save("./weight/w.neuro.json")
