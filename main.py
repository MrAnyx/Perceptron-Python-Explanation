import random
import math
import numpy as np
import matplotlib.pyplot as plt

from Perceptron import Perceptron


# AND operand
inputs_training = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs_training = [0, 0, 0, 1]

p = Perceptron(inputs_training, outputs_training, 10_000)
p.train()

print(p.guess([0, 1]))
