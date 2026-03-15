import pandas as pd
import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(torch.Size(x))
print(x.numel())
x = x.reshape(3, 4)
print(x)
zeros = torch.zeros((2, 3, 4))
print(zeros)

oness = torch.ones((2, 3, 4))
print(oness)
ar = torch.randn(3, 4)
print(ar)
t = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
print()
print(t)

x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([5, 6, 7, 8])
print(x + y, x - y, x * y, x / y, x / y, x ** y, x % y, x // y, x // y, x % y)
a = torch.exp(x)
print()
print(a)

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
r = torch.cat((X, Y), 0)
s = torch.cat((X, Y), 1)
print()
print(r, s)
print()

print(X.sum())

# 2.1.3 Mecanismo de Broadcasting

a = torch.arange(3, dtype=torch.float32).reshape((3 , 1))
b = torch.arange(2, dtype=torch.float32).reshape((1 , 2))
print()
print(a, b)
print()
print(a  + b)

A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

import os
os.makedirs(os.path.join('data'), exist_ok=True)
data_file = os.path.join('data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # Column names
    f.write('NA,Pave,127500\n') # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


# leitura
data = pd.read_csv(data_file)

print(data)
print(data.dtypes)

# separação
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# preencher apenas numérico
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())

print("\nApós fillna numérico:")
print(inputs)

# one hot encoding
inputs = pd.get_dummies(inputs, dummy_na=True)

print("\nApós get_dummies:")
print(inputs)

# converter tudo para float
X = torch.tensor(inputs.astype(float).values)
y = torch.tensor(outputs.values)

print("\nTensor X:")
print(X)

print("\nTensor y:")
print(y)