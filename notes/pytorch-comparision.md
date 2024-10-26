
In PyTorch, everything is defined in Tensors.
Tensors are just n-dimensional array of scalars.

```
import torch

x1 = torch.Tensor([2.0]).double() ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double() ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double() ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double() ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double() ; b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())
```

Now, in PyTorch these values are stored as Tensor arrays

For example: 
```
torch.Tensor([[1,2,3], [4,5,6]])
```
```
torch.Tensor([[1,2,3], [4,5,6]]).shape
```

Now for the datatype, python by default gives double to its decimal numbers.

But PyTorch gives it only a single precision float i.e. Float32:
```
torch.Tensor([2.0]).dtype
```

Therefore we double it by adding:
```
torch.Tensor([2.0]).double().dtype
```
And we get Float64


Also, PyTorch assumes that we do not need the gradient values of the leaf nodes, so we manually assign them there - x1.requires_grad = True.
By default they are set to False (For efficiency reasons)

Now, we have made the variable declarations based on PyTorch.

When we try to access the values individually, they will be stored inside tensor objects
```
x2.data
```

In order to remove the tensor wrap, we just add .item() to it
```
x2.data.item()
```

--------

So the idea was to prove that everything we've build till now very much agrees with the PyTorch API syntax. 
But to also realize that everything will become just for efficient with PyTorch.

```embed
title: "NeuralNetworks-Backpropagation/x10_implementing_in_pytorch.ipynb at main · MuzzammilShah/NeuralNetworks-Backpropagation"
image: "https://opengraph.githubassets.com/b5b980143a1ada8850eb5544301e703e026969a4fa82cd4031dbe2582f294473/MuzzammilShah/NeuralNetworks-Backpropagation"
description: "[ 2nd October, 2024 - PRESENT ]. Contribute to MuzzammilShah/NeuralNetworks-Backpropagation development by creating an account on GitHub."
url: "https://github.com/MuzzammilShah/NeuralNetworks-Backpropagation/blob/main/x10_implementing_in_pytorch.ipynb"
```
