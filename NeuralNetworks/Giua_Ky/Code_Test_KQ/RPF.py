import numpy as np

class RBF:
    def __init__(self, In_num=1, Out_num=1, Φ_num =1):
        self.w = np.array([[0.1], [0.3]])
        self.Φ_num = Φ_num
        self.Out_num = Out_num
        self.In_num = In_num

    def _step(self, net):
        if net >= 0:
            return 1
        elif net < 0:
            return 0
        y = net
        return y
    
    # def tinhKGC(seft, x, u):
    #     return np.exp(-np.linalg.norm(x-u)**2)

    # def _Φ(seft, num, x, u):
    #     Φ = np.zeros(num)
    #     for i in range(num):
    #         Φ[i] = np.exp(-np.linalg.norm(x-u[i])**2) 
    #     return Φ
    
    # def _In(self, x, u ):
    #     P = np.zeros((x.shape[0], self.Φ_num))
    #     for i in range(x.shape[0]):
    #         P[i] = self._Φ(self.Φ_num, x[i], u)
    #     return P

    def _Φ(self, x, u, σ):
        Φ = np.zeros((x.shape[0], self.Φ_num))
        for i in range(x.shape[0]):
            for j in range(self.Φ_num):
                Φ[i, j] = np.exp(-np.linalg.norm(x[i]-u[j])**2 / (2*σ**2))
        return Φ


    def training(self, X, U, D, eta,σ, max_epoch):   
        Φ = np.array(self._Φ(X, U, σ)).T
        print(Φ.T)

        E = 1
        epoch = 0
        input_number, data_number = Φ.shape

        while epoch < max_epoch:
            E = 0
            
            for i, Φ in enumerate(Φ.T):
                Φ = Φ.reshape(input_number, 1)
                net = self.w.T @ Φ
                y = self._step(net)
                
                self.w += (eta * (D[i] - y) * Φ)

                E += 1/2 * (D[i] - y)**2 

                # print(f"\nk = {i+1}")
                # print("y =", y)
                # print("W =\n", self.w)
                # print(f'E = {E}')

            epoch += 1
            if E == 0:
                break
            
            print(f'\nEpoch: {epoch}, Error: {E}')
            print("____________________")
            
if __name__ == "__main__":
    nn = RBF(3, 1, 2)
    X = np.array([[0, 0,], [0, 1], [1, 1]])

    U = np.array([[1, 1], [0, 0]])

    D = np.array([0, 0, 1])


    nn.training(X, U, D, 0.3, 0.5, 3)