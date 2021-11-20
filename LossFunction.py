import math


class LossFunction:
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    HEAVISIDE = "heaviside"

    @staticmethod
    def heaviside(x):
        return 1 if x >= 0 else 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def tanh(x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    @staticmethod
    def relu(x):
        return x if x >= 0 else 0

    @staticmethod
    def loss_methods_switch_process():
        return {
            LossFunction.RELU: LossFunction.relu,
            LossFunction.TANH: LossFunction.tanh,
            LossFunction.SIGMOID: LossFunction.sigmoid,
            LossFunction.HEAVISIDE: LossFunction.heaviside,
        }
