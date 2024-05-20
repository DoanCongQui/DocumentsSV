import numpy as np

class CNN:
    def __init__(selfl):
        pass

    def convolution(self, image, kernel, stride, padding=0):
        image = np.pad(image, (padding, padding), mode='constant', constant_values=0)
        height, width = image.shape[:2]
        kernel_height, kernel_width = kernel.shape

        result_height = (height - kernel_height) // stride + 1
        result_width = (width - kernel_width) // stride + 1
        
        result = np.zeros((result_height, result_width))
        for i in range(0, result_height):
            for j in range(0, result_width):
                window = image[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
                result[i, j] = np.sum(np.multiply(window, kernel))
        return result
    
    def convolution_max(self, image, kernel_size, stride, padding=0):
        image = np.pad(image, (padding, padding), mode='constant', constant_values=0)
        height, width = image.shape[:2]

        result_height = (height - kernel_size[0]) // stride + 1
        result_width = (width - kernel_size[1]) // stride + 1

        result = np.zeros((result_height, result_width))

        for i in range(0, result_height):
            for j in range(0, result_width):
                window = image[i*stride:i*stride+kernel_size[0], j*stride:j*stride+kernel_size[1]]
                result[i, j] = np.max(window) 
        return result
    
if __name__ == "__main__":
    image = np.array([
        [1, 2, 3, 4, 5],
        [6, 7.7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])

    kernel = np.array([
        [7, 10],
        [1, 1]
    ])
    
    cnn = CNN()
    x = cnn.convolution_max(image, (2, 2), 1, padding=0)  

    print(x)

    