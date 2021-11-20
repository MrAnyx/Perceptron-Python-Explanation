from Perceptron import Perceptron


# OR operand
inputs_training = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs_training = [0, 1, 1, 1]

p = Perceptron(inputs_training, outputs_training, 10_000)
p.train()

print(p.guess([0, 0]))
