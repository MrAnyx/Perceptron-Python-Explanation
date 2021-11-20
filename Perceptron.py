import numpy as np


class Perceptron:
    def __init__(self, inputs_training, outputs_training, nb_iter):
        self.inputs_training = inputs_training
        self.outputs_training = outputs_training
        self.nb_iter = nb_iter
        self.training_factor = 0.0001
        self.bias = 0
        self.nb_feature = len(self.inputs_training[0])
        self.weights = [self.bias] + [0 for i in range(self.nb_feature)]

    def heaviside(self, x):
        return 1 if x >= 0 else 0

    def format_input_bias(self, input, value):
        return [value] + input

    def train(self):
        nb_inputs = len(self.inputs_training)
        for _ in range(self.nb_iter):
            for i in range(nb_inputs):
                sum_val = 0
                inputs_tmp = np.array(
                    self.format_input_bias(self.inputs_training[i], 1)
                )

                sum_val = np.sum(np.multiply(inputs_tmp, self.weights))
                output = self.heaviside(sum_val)

                for j in range(self.nb_feature + 1):
                    self.weights[j] += (
                        self.training_factor
                        * (self.outputs_training[i] - output)
                        * inputs_tmp[j]
                    )

    def guess(self, input):
        inputs_tmp = np.array(self.format_input_bias(input, 1))

        sum_val = np.sum(np.multiply(inputs_tmp, self.weights))
        return self.heaviside(sum_val)
