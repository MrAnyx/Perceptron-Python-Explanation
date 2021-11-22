import numpy as np
import matplotlib.pyplot as plt
from LossFunction import LossFunction
import random


class Perceptron:
    def __init__(
        self,
        inputs_training,
        outputs_training,
        nb_iter,
        loss_func,
        weights=None,
    ) -> None:
        self.inputs_training = inputs_training
        self.outputs_training = outputs_training
        self.nb_iter = nb_iter
        self.training_factor = 0.01
        self.bias = 0
        self.nb_feature = len(self.inputs_training[0])

        if weights:
            self.weights = weights
        else:
            self.weights = [self.bias] + [0 for i in range(self.nb_feature)]

        self.loss_func = loss_func
        self.activation_function = LossFunction.loss_methods_switch_process()[
            self.loss_func
        ]
        self.nb_inputs = len(self.inputs_training)
        self.E = []

    def format_input_bias(self, input, value) -> list:
        return [value] + input

    def train(self, display_loss=True) -> None:
        for k in range(self.nb_iter):
            output_list = []
            for i in range(self.nb_inputs):
                sum_val = 0
                inputs_tmp = np.array(
                    self.format_input_bias(self.inputs_training[i], 1)
                )

                sum_val = np.sum(np.multiply(inputs_tmp, self.weights))
                output = self.activation_function(sum_val)
                output_list.append(output)

                for j in range(self.nb_feature + 1):
                    self.weights[j] += (
                        self.training_factor
                        * (self.outputs_training[i] - output)
                        * inputs_tmp[j]
                    )

            error = self.compute_error(self.outputs_training, output_list)
            self.E.append(error)
            # if display_loss and k % (self.nb_iter / 100) == 0:
            if display_loss and self._compute_iter_display_error(k):
                print(f"Pourcentage d'erreur ({k}) : {error} -> {round(error*100, 2)}%")

    def _compute_iter_display_error(self, iter) -> bool:

        if iter % (10 ** (max(0, len(str(self.nb_iter)) - 2))) == 0:
            return True
        else:
            return False

    def compute_error(self, target, output) -> float:
        sum_tmp = 0
        for i in range(self.nb_inputs):
            sum_tmp += (output[i] - target[i]) ** 2

        return (1 / self.nb_inputs) * sum_tmp

    def display_mean_squared_loss_graph(self) -> None:
        plt.plot(range(self.nb_iter), self.E)
        plt.xlabel("Itération")
        plt.ylabel("Pourcentage d'erreur")
        plt.title(f"Taux d'erreur avec la fonction d'activation {self.loss_func}")
        plt.show()

    def guess(self, input) -> float:
        inputs_tmp = np.array(self.format_input_bias(input, 1))

        sum_val = np.sum(np.multiply(inputs_tmp, self.weights))
        return self.activation_function(sum_val)
