import numpy as np

class Perceptron:
    def __init__(self, In_num=1, Out_num=1):
        self.w = np.array([[0.5], [-0.7], [1]]) 
        # self.w = np.random.rand(Hidden_num, Out_num)
        self.Out_num = Out_num

    def _step(self, net):
        if net >= 0:
            return 1
        elif net < 0:
            return 0
        y = net
        return y

    def training(self, X, D, eta, max_epoch):
        E = 1
        epoch = 0
        input_number, data_number = X.shape

        while epoch < max_epoch:
            E = 0

            for i, x in enumerate(X.T):
                x = x.reshape(input_number, 1)
                net = self.w.T @ x
                y = self._step(net)

                self.w += (eta * (D[i] - y) * x)

                E += 1/2 * (D[i] - y)**2 


                print(f"\n\t| k = {i+1} |\n\t+-------+")
                print(net)
                print("y =", y)
                print("W =\n", self.w)
                print(f'E = {E}')
                print("_________")

            epoch += 1
            if E == 0:
                break
            
            print(f'\nEpoch: {epoch}, Error: {E}')
            print("____________________")
            
            
if __name__ == "__main__":
    nn = Perceptron(3, 1)
    X = np.array([[0.8147, 0.095, 0.127, 0.9123, 0.6324],
                [0.095, 0.2785, 0.5669, 0.9575, 0.9649],
                [1, 1, 1, 1, 1]])
    
    D = np.array([0, 0, 0, 1, 1])

    nn.training(X, D, 0.33, 3)