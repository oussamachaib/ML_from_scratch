import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy.io import loadmat

#%% To-do list
# Convert to class
# Add more than one layer compatibility
# Clean up & annotate

#%% Loading actual MNIST dataset
mnist = loadmat("mnist-original.mat")
X = mnist["data"].T
Y = mnist["label"][0].astype(int)

#%% Splitting the data
N = X.shape[0]
ids = np.arange(0, N)
shuffle(ids)
X = X[ids,:]
Y = Y[ids]

X_train = X[0:50000,:]
X_test = X[50000:,:]
y_train = Y[0:50000]
y_test = Y[50000:]

#%% Plotting a sample
idx = 3
img = X_train[idx,:].reshape(28,28)

plt.figure(figsize = (3,3));
plt.imshow(img, cmap = 'gray');
print(f'Label: {y_train[idx]}')
plt.show()

#%% Defining activation functions

def ReLU(x):
    return np.maximum(x,0)

def softmax(y):
    y_shifted = y - np.max(y, axis=1, keepdims=True)
    exp_y = np.exp(y_shifted)
    return exp_y / np.sum(exp_y, axis=1, keepdims=True)


# def softmax_deriv(y):
#     dsigma = softmax(y)*(1-softmax(y))
#     #dsigma[dsigma < 1e-15] = 1e-15
#     return dsigma

def sigmoid(y):
    return 1/(1+np.exp(-y))

def sigmoid_deriv(y):
    return sigmoid(y)*(1-sigmoid(y))

def ReLU_deriv(x):
    return x>0

#%% Forward pass

alpha = 0.1
n_batch = 100
units = 100
N_train = len(y_train)

L = []
acc = []

# Batching (1 sample/batch)
for i in range(n_batch):
    X_batch = X_train[int((N_train/n_batch)*i) : int(((N_train)/n_batch)*(i+1)),:]
    y_batch = y_train[int((N_train/n_batch)*i) : int(((N_train)/n_batch)*(i+1))]

    if i == 0:
        k = len(np.unique(y_train))
        n, p = X_batch.shape
        # W1 = np.random.randn(p, units)*0.01
        b1 = np.zeros((1, units))
        # W2 = np.random.randn(units, k)*0.01
        b2 = np.zeros((1, k))
        W1 = np.random.randn(p, units) * np.sqrt(2 / p)
        W2 = np.random.randn(units, k) * np.sqrt(2 / units)

    # Layer 1
    Z1 = np.dot(X_batch,W1) + b1
    A1 = sigmoid(Z1)

    # Output layer
    Z2 = np.dot(A1,W2) + b2
    A2 = softmax(Z2) # predicted probabilities (y_hat)

    # Loss function
    y = np.eye(N = n, M = k)[y_batch]

    # Computing loss function (to track model learning)
    y_predict = np.argmax(A2, axis = 1, keepdims=True)[:,0]
    accuracy = (1/n)*np.sum(y_predict == y_batch)
    y_predict = np.eye(N = n, M = k)[y_predict].copy()
    A2_copy = A2.copy()
    A2_copy[A2_copy<1e-15] = 1e-15
    loss = - (1/n)*np.sum((y*np.log(A2_copy)))

    # Backpropagation : Output
    dL_dZ2 = A2-y
    dZ2_dW2 = A1
    dZ2_db2 = np.ones((n,1))
    dL_dW2 = np.dot(dZ2_dW2.T,dL_dZ2)/n
    dL_db2 = np.dot(dZ2_db2.T,dL_dZ2)/n

    # Backpropagation : Input
    dL_dZ1 = np.dot(dL_dZ2, W2.T)*sigmoid_deriv(Z1)
    dZ1_dW1 = X_batch
    dZ1_db1 = np.ones((n,1))
    dL_dW1 = np.dot(dZ1_dW1.T,dL_dZ1)/n
    dL_db1 = np.dot(dZ1_db1.T,dL_dZ1)/n

    # Gradient descent -- Updating network parameters
    W2 -= alpha*dL_dW2
    b2 -= alpha*dL_db2
    W1 -= alpha*dL_dW1
    b1 -= alpha*dL_db1

    print(f'Epoch {i}')
    print(f'Training loss: {loss:.2}')
    print(f'Training accuracy: {accuracy:.2}\n\n')

    L.append(loss)
    acc.append(accuracy)

    #break

#%

plt.figure(dpi = 300)
plt.plot(L/L[0], '-')
plt.plot(acc)
plt.xlabel('Epoch')
plt.ylabel('Train loss/accuracy')
plt.show()

#% Test accuracy

# Layer 1
Z1 = np.dot(X_test,W1) + b1
A1 = sigmoid(Z1)
# Output layer
Z2 = np.dot(A1,W2) + b2
A2 = softmax(Z2) # predicted probabilities (y_hat)
# Computing accuracy
y_predict = np.argmax(A2, axis = 1, keepdims=True)[:,0]
accuracy = (1 / len(y_predict)) * np.sum(y_predict == y_test)

print(f'Test accuracy: {accuracy:.2}\n\n')
