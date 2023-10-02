import math
import random
import json

activation = math.tanh
deactivation = math.tanh


class NeuralNetwork:
    def __init__(self, neurons_size: list, weights: list = None):
        # Количество слоёв
        self.neuron_size = neurons_size

        # добавление нейрона смещения на каждый слой, кроме out
        for i in range(len(neurons_size) - 1):
            self.neuron_size[i] += 1

        # если веса не заданы
        if weights is None:
            self.weights = []
            # создаем случайные веса
            for i in range(len(neurons_size) - 1):
                self.weights.append(
                    [
                        [random.uniform(-1, 1) for x in range(neurons_size[i + 1])]
                        for y in range(neurons_size[i])
                    ]
                )
        else:
            self.weights = weights

    def out(self, inp):
        # + [1] - это добавление значения для нейрона смещения, про который нет надобности помнить при вводе параметров
        out = [inp + [1]]
        for i in range(1, len(self.weights) + 1):
            a = []
            for j in range(self.neuron_size[i]):
                s = sum(
                    [
                        out[i - 1][k] * self.weights[i - 1][k][j]
                        for k in range(self.neuron_size[i - 1])
                    ]
                )
                a.append(activation(s))
            if i != len(self.weights):
                a += [1]
            out.append(a)
        return out

    def correct(self, inp, answer, learning_rate=0.1):
        out = self.out(inp)
        errors = [[answer[i] - out[-1][i] for i in range(len(out[-1]))]]

        # считаем ошибку каждого нейрона
        for i in range(len(self.weights) - 1, 0, -1):
            a = []
            for j in range(self.neuron_size[i]):
                s = sum(
                    [
                        errors[0][k] * self.weights[i][j][k]
                        for k in range(self.neuron_size[i + 1])
                    ]
                )
                # корректируем ошибку с производной функции активации(если у вас не tanh - измените)
                a.append((1 - out[i][j] ** 2) * s)
            errors.insert(0, a)

        # обновляем веса
        for i in range(len(self.weights)):
            for j in range(self.neuron_size[i]):
                for k in range(self.neuron_size[i + 1]):
                    self.weights[i][j][k] += learning_rate * errors[i][k] * out[i][j]

        error_count = sum([sum(abs(en) for en in el) for el in errors])
        return out, error_count

    def save(self, name):
        with open(name, "w") as f:
            neuron_size = self.neuron_size.copy()
            for i in range(len(neuron_size) - 1):
                neuron_size[i] -= 1
            f.write(json.dumps({"shape": neuron_size, "weights": self.weights}))

    def open(name):
        with open(name, "r") as f:
            data = json.loads(f.read())
            return NeuralNetwork(data["shape"], data["weights"])
