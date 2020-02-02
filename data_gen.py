import numpy as np

class Data():
    def __init__(self, data_path, n_test_data=3000):
        '''Data generation
        x_* attributes contain 14x14 MNIST images of digits 0,2,4,6,8
        y_* attributes are the digit labels'''
        
        data = np.loadtxt(data_path)
        np.random.shuffle(data)
        data_length = len(data)
        y = np.zeros((data_length,1,5))
        x = np.zeros((data_length,1,14,14))
        
        for i in range(data_length):
            data_i = data[i]
            index = int(data_i[-1]/2)
            y[i,0,index] = 1
            x[i,0,:,:] = np.reshape(data_i[:-1],(14,14))
        
        x_train= x[n_test_data:,:,:]
        y_train= y[n_test_data:]
        x_train= np.array(x_train, dtype= np.float32)
        y_train= np.array(y_train, dtype= np.float32)

        x_test= x[:n_test_data,:,:]
        y_test= y[:n_test_data]
        x_test= np.array(x_test, dtype= np.float32)
        y_test= np.array(y_test, dtype= np.float32)

        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test