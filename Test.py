import torch

x = torch.rand(5,3)
print(x)

print(torch.cuda.is_available())
def f (x,l=[]):
    for i in range(x):
        l.append(i*i)
    print(l)

f(2)
f(3,[3,2,1])



