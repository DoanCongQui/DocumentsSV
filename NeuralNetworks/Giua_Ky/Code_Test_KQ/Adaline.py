import numpy as np

class Adaline:
    def __init__(self, In_num=1, Out_num=1):
        self.w = np.array([[0.5], [-0.7], [1]]) 
        # self.w = np.random.rand(Hidden_num, Out_num)
        self.Out_num = Out_num

    def training(self, X, D, eta, max_epoch):
        E = 1
        epoch = 0
        input_number, data_number = X.shape

        while epoch < max_epoch:
            E = 0

            for i, x in enumerate(X.T):
                x = x.reshape(input_number, 1)
                net = self.w.T @ x
                y = net

                self.w += (eta * (D[i] - y) * x)

                E += 1/2 * (D[i] - y)**2 

                print(f"\nk = {i+1}")
                print("y =", y)
                
                print(f'E = {E}')

            epoch += 1
            if E == 0:
                break
            print("W =\n", self.w)
            print(f'Epoch: {epoch}, Error: {E}')
            print("_________________________\n")
            
            
if __name__ == "__main__":
    nn = Adaline(3, 1)
    X = np.array([[0.8147, 0.095, 0.127, 0.9123, 0.6324],
                [0.095, 0.2785, 0.5669, 0.9575, 0.9649],
                [1, 1, 1, 1, 1]])
    
    D = np.array([0, 0, 0, 1, 1])

    nn.training(X, D, 0.33, 3)