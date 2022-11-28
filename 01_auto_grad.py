import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2 # <AddBackward0>
print(y)

z = y*y*2 # <MulBackward0>
# z = z.mean() # <MeanBackward0>
print(z)

# dz/dx，如果一开始没有设置 requires_grad 为 true，那么这一步就会报错
# 如果 z 没有做 z.mean() 这一步，那么 z 不是一个 scalar，就不能够做 backward
# 除非，在 backward 的参数中传入一个 vector
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
print(x.grad) # calculate x's gradient

# 将 x 改为 requires_grad 为 false
x.requires_grad_(False) # 方法一
x_new = x.detach() # 方法二
with torch.no_grad(): # 方法三
  y = x + 2
  print(y) # 去掉了 gradient function 部分


weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad) # tensor([3., 3., 3., 3.]) -> tensor([6., 6., 6., 6.]) -> tensor([9., 9., 9., 9.])

    weights.grad.zero_() # all are tensor([3., 3., 3., 3.]) 梯度清零

# Do the same thing
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()
