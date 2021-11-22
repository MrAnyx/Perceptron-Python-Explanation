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
nb_iter = 2_000

p = Perceptron(inputs_training, outputs_training, nb_iter, LossFunction.RELU)
p.train(display_loss=False)
p.display_mean_squared_loss_graph()


# prob = p.guess([0, 0])
# print(prob)
# print(p.weights)
