from Perceptron import Perceptron
from LossFunction import LossFunction

# OR gate
inputs_training = [
    [0, 0, 0],  # 0
    [0, 0, 1],  # 1
    [0, 1, 0],  # 1
    [0, 1, 1],  # 1
    [1, 0, 0],  # 1
    [1, 0, 1],  # 1
    [1, 1, 0],  # 1
    [1, 1, 1],  # 1
]
outputs_training = [0, 1, 1, 1, 1, 1, 1, 1]
nb_iter = 5_000

p = Perceptron(inputs_training, outputs_training, nb_iter, LossFunction.SIGMOID)
p.train(display_loss=True)
p.display_mean_squared_loss_graph()

prob = p.guess([0, 0, 0])
print(prob)
