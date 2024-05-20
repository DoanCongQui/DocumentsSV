import numpy as np
import matplotlib.pyplot as plt

class BackproPagation:
    def __init__(self, In_num=1, Out_num=1, Hidden_num=1):
        # self.w = np.array([[-0.2], [0.3]]) 
        # self.v = np.array([[0.7, 0.6], [-0.8, 0.5]])
        self.w = np.random.rand(Hidden_num, Out_num)
        self.v = np.random.rand(In_num, Hidden_num)
        self.hidden_num = Hidden_num
        self.Out_num = Out_num

    def _activation_function(self, net):
        y = (1 - np.exp(-2 * net)) / (1 + np.exp(-2 * net))
        return y

    def cal_output(self, X):
        net_h = self.v.T @ X
        yh = self._activation_function(net_h)
        net_o = self.w.T @ yh
        y = net_o
        return y

    def training(self, X, D, eta, max_epoch):
        E = 1
        epoch = 0
        input_number, data_number = X.shape

        while epoch < max_epoch:
            E = 0
            epsilon=1e-5

            for i, x in enumerate(X.T):
                x = x.reshape(input_number, 1)
                net_h = self.v.T @ x
                yh = self._activation_function(net_h)

                net_o = self.w.T @ yh
                y = net_o

                error = D[i] - y
                E += 1/2 * (error)**2 
                delta_o = error

                w_old = self.w

                delta_h = (delta_o * w_old) * (1 - yh**2)
                self.w += eta * delta_o * yh
                
                
                for j in range(self.hidden_num):
                    # self.v[:, j] += eta * delta_h[j] * x.T
                    self.v[:, j] += (eta * delta_h[j] * x.T).reshape(self.v[:, j].shape)

                
                # print("y:", y)
                # print(f'E: {E}')
                # print("W:\n", self.w)
                # print("Delta(h):\n", delta_h)
                # print("V:\n", self.v)  

            epoch += 1
            if E < epsilon:
                break
            print(f'_________________________\nEpoch: {epoch}, Error: {E}\n')
            
if __name__ == "__main__":
    nn = BackproPagation(2, 1, 2)
    X = np.array([[0.3, 0.35, 0.4, 0.8, 0.9, 1, 1.2, 1.6, 2],
                [0.3, 0.4, 0.5, 0.75, 0.7, 0.8, 0.4, 0.5, 0.5]])

    
    D = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])

    print("Cac gia tri da cho")
    print("W:\n",nn.w)
    print(f'V:\n {nn.v}\n')

    print("________\nTraining\n________\n")
    nn.training(X, D, 0.3, 100)

    X_test = np.array([[0.35, 0.35, 0.85],
                       [0.35, 0.7, 0.45]])
            
    # y_predicted = nn.cal_output(X)
    # plt.plot(range(len(D)), D, label='Thuc te')
    # plt.plot(range(len(y_predicted)), y_predicted, label='Du doan')
    # plt.xlabel('Mau du lieu')
    # plt.ylabel('Gia tri dau ra')
    # plt.title('So sanh gia tri thuc te va gia tri du doan')
    # plt.legend()
    # plt.show()
            