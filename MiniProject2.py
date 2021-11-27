#!/usr/bin/env python
# coding: utf-8

# # Comp551 Mini Project2

# ## Part1 - Step1

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import display
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import timeit
import random
random.seed(551)


# In[ ]:


def cost_fn(x, y, w):
    N, D = x.shape                                                       
    z = np.dot(x, w)
    J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)))  #log1p calculates log(1+x) to remove floating point inaccuracies 
    return J


# In[ ]:


logistic = lambda z: 1./ (1 + np.exp(np.array(-z, dtype=np.float128)))       #logistic function
def gradient(x, y, w):
    N,D = x.shape
    yh = logistic(np.dot(x, w))    # predictions  size N
    grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
    return grad                         # size D

def gradient_descent(x, y, w, epsilon, learning_rate, max_iters):
    N,D = x.shape
    grad = np.inf
    t = 1
    cost_t = []
    cost_v = []
    # np.linalg.norm(grad) > epsilon or
    while t <= max_iters:
        grad = gradient(x, y, w)               # compute the gradient with present weight
        w = w - learning_rate * grad         # weight update step
        t += 1
        cost_t.append(cost_fn(x, y, w))
        Nt = validation_features.shape[0]
        cost_v.append(cost_fn(np.column_stack([validation_features,np.ones(Nt)]), validation_labels, w))
    return w, grad, t, cost_t, cost_v


# In[ ]:


class LogisticRegression:
    
    def __init__(self, gd_func, add_bias=True, learning_rate=.1, epsilon=1e-4, max_iters=1e5, verbose=False, batch_size=None, decay_rate=None):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients 
        self.max_iters = max_iters                    #maximum number of iteration of gradient descent
        self.verbose = verbose
        self.gd_func = gd_func
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        w0 = np.zeros(D)
        if self.decay_rate != None:
            self.w, g, t, cost_t, cost_v = self.gd_func(x=x, y=y, w=w0, epsilon=self.epsilon, learning_rate=self.learning_rate, max_iters=self.max_iters, batch_size=self.batch_size, decay_rate=self.decay_rate)
        elif self.batch_size != None:
            # self.w, g, t, train, val = self.gd_func(x=x, y=y, w=w0, epsilon=self.epsilon, learning_rate=self.learning_rate, max_iters=self.max_iters, batch_size=self.batch_size)
            self.w, g, t, cost_t, cost_v = self.gd_func(x=x, y=y, w=w0, epsilon=self.epsilon, learning_rate=self.learning_rate, max_iters=self.max_iters, batch_size=self.batch_size)
        else:
            self.w, g, t, cost_t, cost_v = self.gd_func(x=x, y=y, w=w0, epsilon=self.epsilon, learning_rate=self.learning_rate, max_iters=self.max_iters)
        
        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
        # return np.linalg.norm(g), train, val
        return np.linalg.norm(g), cost_t, cost_v
        # return np.linalg.norm(g)
    
    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))            #predict output
        return yh


# In[65]:


d_train = pd.read_csv("diabetes_train.csv")

train_features = normalize(d_train.drop(['Outcome'],axis=1), norm='l2')
train_labels = d_train['Outcome']

d_test = pd.read_csv("diabetes_test.csv")
test_features = normalize(d_test.drop(['Outcome'],axis=1), norm='l2')
test_labels = d_test['Outcome']

d_val = pd.read_csv("diabetes_val.csv")
validation_features = normalize(d_val.drop(['Outcome'],axis=1), norm='l2')
validation_labels = d_val['Outcome']


# In[67]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(max_iters=50000, learning_rate=0.1, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(50000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Batch Size 600 and Without Momentum")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[26]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=1, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 1.0")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[36]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.001, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.001")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[37]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.01, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.01")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[38]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.1, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.1")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[39]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.2, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.2")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[40]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.3, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.3")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[41]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.4, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.4")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[32]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.5, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.5")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[31]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.6, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.6")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[30]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.7, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.7")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[29]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.8, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.8")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[28]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(learning_rate=0.9, verbose=True, gd_func=gradient_descent)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(100000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Leaning Rate 0.9")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# ## Part1 Step2

# In[68]:


logistic = lambda z: 1./ (1 + np.exp(np.array(-z, dtype=np.float128)))       #logistic function
def cost_fn(x, y, w):
    N, D = x.shape                                                       
    z = np.dot(x, w)
    J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)))  #log1p calculates log(1+x) to remove floating point inaccuracies 
    return J
def gradient(x, y, w):
    N,D = x.shape
    yh = logistic(np.dot(x, w))    # predictions  size N
    grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
    return grad                         # size D
def stochastic_gradient_descent(x, y, w, epsilon, learning_rate, max_iters, batch_size):
    x, y = np.array(x), np.array(y)
    n_obs = x.shape[0]
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
    rng = np.random.default_rng(seed=551)
    grad = np.inf
    t = 1
    max_iters = int(max_iters)
    learning_rate = np.array(learning_rate)
    cost_t = []
    cost_v = []
    for i in range(max_iters):
        t = i + 1
        rng.shuffle(xy)
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
            y_batch  = y_batch.flatten()
            grad = gradient(x_batch, y_batch, w)
            # To ensure the sgd can converge
            diff = - learning_rate * grad
            w += diff
            # if np.all(np.abs(diff) <= 1e-06):
                # break
        # if np.linalg.norm(grad) < epsilon:
            # break
        cost_t.append(cost_fn(x, y, w))
        Nt = validation_features.shape[0]
        cost_v.append(cost_fn(np.column_stack([validation_features,np.ones(Nt)]), validation_labels, w))
    return w, grad, t, cost_t, cost_v


# In[69]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(max_iters=50000, learning_rate=0.1, verbose=True, gd_func=stochastic_gradient_descent, batch_size=15)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)


# In[70]:


x = np.arange(50000)
# print(cost_t)
# print(cost_v)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Batch Size 15")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[71]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(max_iters=50000, learning_rate=0.1, verbose=True, gd_func=stochastic_gradient_descent, batch_size=30)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(50000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Batch Size 30")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[72]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(max_iters=50000, learning_rate=0.1, verbose=True, gd_func=stochastic_gradient_descent, batch_size=60)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(50000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Batch Size 60")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[73]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(max_iters=50000, learning_rate=0.1, verbose=True, gd_func=stochastic_gradient_descent, batch_size=120)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(50000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Batch Size 120")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[74]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(max_iters=50000, learning_rate=0.1, verbose=True, gd_func=stochastic_gradient_descent, batch_size=300)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(50000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Batch Size 300")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# In[75]:


training_error = []
validation_error = []
gs = []
time = []
size = [1, 2, 4, 8,15,30,60,120,300, 600]
model = LogisticRegression(max_iters=50000, learning_rate=0.1, verbose=True, gd_func=stochastic_gradient_descent, batch_size=600)
start = timeit.default_timer()
result, cost_t, cost_v = model.fit(train_features, train_labels)
stop = timeit.default_timer()
time.append(stop - start)
gs.append(result) 
train_prediction = model.predict(train_features)
validation_prediction = model.predict(validation_features)
mse_train = mean_squared_error(train_labels, train_prediction)
mse_validation = mean_squared_error(validation_labels, validation_prediction)
training_error.append(mse_train)
validation_error.append(mse_validation)
x = np.arange(50000)
plt.plot(x, cost_t, label='Training Cost')
plt.plot(x, cost_v, label="Validation Cost")
# plt.plot(x, gs, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title("Epoch &  Cost With Batch Size 600")
plt.legend()
plt.show()
print(mse_train)
print(mse_validation)
minimum = min(cost_v)
print("Min of Validation Cost " + str(minimum))
print("Got min on iteration " + str(cost_v.index(minimum)))


# ## Part1 Step3

# In[108]:


logistic = lambda z: 1./ (1 + np.exp(np.array(-z, dtype=np.longdouble)))       #logistic function
def gradient(x, y, w):
    N,D = x.shape
    yh = logistic(np.dot(x, w))    # predictions  size N
    grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
    return grad                         # size D

def predict(x, w):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        x = np.column_stack([x,np.ones(Nt)])
        
        yh = logistic(np.dot(x, w))            #predict output
        return [1 if y > 0.5 else 0 for y in yh]

def momentum_SGD(x, y, w, epsilon, learning_rate, max_iters, batch_size, decay_rate):
    n_obs = x.shape[0]
    x, y = np.array(x), np.array(y)
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
    rng = np.random.default_rng(seed=551)
    grad = np.inf
    t = 1
    decay_rate = np.array(decay_rate)
    max_iters = int(max_iters)
    learning_rate = np.array(learning_rate)
    diff = 0
    
    #train_accuracy = []
    #validation_accuracy = []
    cost_t = []
    cost_v = []
    for i in range(max_iters):
        t = i + 1
        rng.shuffle(xy)
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
            y_batch  = y_batch.flatten()
            grad = gradient(x_batch, y_batch, w)
            # To ensure the sgd can converge
            diff = decay_rate * diff + (1 - decay_rate) * grad
            if np.all(np.abs(diff) <= 1e-06):
                break
            w += (-learning_rate * diff)
        
        cost_t.append(cost_fn(x, y, w))
        Nt = validation_features.shape[0]
        cost_v.append(cost_fn(np.column_stack([validation_features,np.ones(Nt)]), validation_labels, w))
        """
        train_prediction = predict(train_features, w)
        validation_prediction = predict(validation_features, w)
        accuracy_train = accuracy_score(train_labels, train_prediction)
        accuracy_validation = accuracy_score(validation_labels, validation_prediction)
        train_accuracy.append(accuracy_train)
        validation_accuracy.append(accuracy_validation)
        """
    return w, grad, t, cost_t, cost_v    
    #return w, grad, t, train_accuracy, validation_accuracy


# In[174]:


def momentum_result(iters, batch_size, momentum):
    model = LogisticRegression(max_iters = iters,learning_rate=0.001, verbose=True, gd_func=momentum_SGD, batch_size=batch_size,decay_rate=momentum)
    g, train, val = model.fit(train_features, train_labels)
    x = np.arange(iters)
    plt.plot(x, train, label='Training Cost')
    plt.plot(x, val, label='Validation Cost')
    # plt.plot(x, gs, label='Gradient')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    title = "Epoch & Cost with batch size = "+str(batch_size)+ ", momentum="+ str(momentum)
    plt.title(title)
    plt.legend()
    plt.show()
    np_val = np.asarray(val,dtype = np.longdouble)
    min_index = np.argmin(np_val)
    print('min cost at epoch', min_index,'min_cost = ',min(val))


# In[161]:


momentum_result(50000,600,0.5)


# In[162]:


momentum_result(50000,600,0.67)


# In[16]:


momentum_result(50000,600,0.75)


# In[17]:


momentum_result(50000,600,0.8)


# In[165]:


momentum_result(50000,600,0.9)


# In[166]:


momentum_result(50000,600,0.99)


# In[167]:


momentum_result(50000,600,0.999)


# ## Part1 Step4

# In[147]:


momentum_result(50000,15,0.5)


# In[148]:


momentum_result(50000,15,0.67)


# In[149]:


momentum_result(50000,15,0.75)


# In[150]:


momentum_result(50000,15,0.8)


# In[151]:


momentum_result(50000,15,0.9)


# In[152]:


momentum_result(50000,15,0.99)


# In[153]:


momentum_result(50000,15,0.999)


# In[154]:


momentum_result(50000,300,0.5)


# In[155]:


momentum_result(50000,300,0.67)


# In[156]:


momentum_result(50000,300,0.75)


# In[157]:


momentum_result(50000,300,0.8)


# In[158]:


momentum_result(50000,300,0.9)


# In[15]:


momentum_result(50000,300,0.99)


# In[160]:


momentum_result(50000,300,0.999)


# ## Part2

# In[1]:


import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from string import punctuation
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier


# In[2]:


def remove_punctuations(sentence):
    for p in punctuation:
        sentence = sentence.replace(p,'')
    return sentence


# In[3]:


def remove_digits(sentence):
    sentence = ''.join([i for i in sentence if not i.isdigit()])
    return sentence


# In[4]:


def Stemmer(text):
    documents = []
    porter = PorterStemmer()
    for s in text:
        s = str(s).lower()
        words_token = word_tokenize(s)
        words = [porter.stem(w) for w in words_token]
        documents.append(' '.join(words))
    return documents


# In[5]:


def lemmatizer(text):
    documents = []
    word_lemmatizer = WordNetLemmatizer()
    for s in text:
        s = str(s).lower()
        words_token = word_tokenize(s)
        words = [word_lemmatizer.lemmatize(w) for w in words_token]
        documents.append(' '.join(words))
    return documents
    


# In[6]:


def stopwords(text):
    stop = stopwords.words('english')
    return ' '.join(word for word in text if word not in stop)


# In[7]:


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression(solver='sag'))])


# In[8]:


text_clf_cv_ngram = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegressionCV(solver='sag',max_iter=1000))])


# In[9]:


text_clf_cv_default = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegressionCV(solver='sag',max_iter=1000))])


# In[25]:


text_clf_MLP = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MLPClassifier(solver='adam',verbose=False, random_state=551))])


# In[12]:


train_df = pd.read_csv("fake_news_train.csv")
train_x = train_df["text"]
train_y = train_df["label"]
val_df = pd.read_csv("fake_news_val.csv")
val_x = val_df["text"]
val_y = val_df["label"]
test_df = pd.read_csv("fake_news_test.csv")
test_x = test_df["text"]
test_y = test_df["label"]


# In[50]:


# accuracy with no data preprocessing 
text_clf_cv_default.fit(train_x, train_y)
val_pred = text_clf_cv_default.predict(val_x)
print("accuracy on val data:", accuracy_score(val_pred, val_y))


# In[52]:


# remove punctuations
train_x_no_punc=train_df["text"].apply(remove_punctuations)
text_clf_cv_default.fit(train_x_no_punc, train_y)
val_pred = text_clf_cv_default.predict(val_x)
print("accuracy on val data:", accuracy_score(val_pred, val_y))


# In[53]:


#remove digits
train_x_no_digits=train_df["text"].apply(remove_digits)
text_clf_cv_default.fit(train_x_no_digits, train_y)
val_pred = text_clf_cv_default.predict(val_x)
print("accuracy on val data:", accuracy_score(val_pred, val_y))


# In[54]:


#Stemming
train_x_stemmer=Stemmer(train_x)
text_clf_cv_default.fit(train_x_stemmer, train_y)
val_pred = text_clf_cv_default.predict(val_x)
print("accuracy on val data:", accuracy_score(val_pred, val_y))


# In[55]:


#Lemmatization
train_x_lemmatizer=lemmatizer(train_x)
text_clf_cv_default.fit(train_x_lemmatizer, train_y)
val_pred = text_clf_cv_default.predict(val_x)
print("accuracy on val data:", accuracy_score(val_pred, val_y))


# In[14]:


# ngram=(1,2)
text_clf_cv_ngram.fit(train_x, train_y)
val_pred = text_clf_cv_ngram.predict(val_x)
print("accuracy on val data:", accuracy_score(val_pred, val_y))


# In[68]:


# ngram=(1,2) and remove punctuations
train_x_no_punc=train_df["text"].apply(remove_punctuations)
text_clf_cv_ngram.fit(train_x_no_punc, train_y)
val_pred = text_clf_cv_ngram.predict(val_x)
print("accuracy on val data:", accuracy_score(val_pred, val_y))


# In[26]:


# MLP
text_clf_MLP.fit(train_x, train_y)
val_pred = text_clf_MLP.predict(val_x)
print("accuracy on val data:", accuracy_score(val_pred, val_y))


# In[16]:


# test
text_clf_cv_ngram.fit(train_x, train_y)
test_pred=text_clf_cv_ngram.predict(test_x)
print("accuracy on test data:", accuracy_score(test_pred, test_y))

