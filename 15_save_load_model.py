# methods:
# torch.save(arg, PATH)
# torch.load(PATH)
# model.load_state_dict(arg)

# # complete model
# torch.save(model, PATH)
# model class must be defined somewhere
# model = torch.load(PATH)
# model.eval()

# # State dict(save the model just for inference)
# torch.save(model.state_dict(), PATH)
# # model must be created again with parameters
# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, n_input_features):
    super(Model, self).__init__()
    self.linear = nn.Linear(n_input_features, 1)
  
  def forward(self, x):
    y_pred = torch.sigmoid(self, self.linear(x))
    return y_pred
  
model = Model(n_input_features=6)

# train your model...

FILE = 'model.pth'

# 两种保存模型的方法：

# method 1: save model directly
torch.save(model, FILE)
model = torch.load(FILE)
model.eval()

for param in model.parameters():
  print(param)

# method 2: save model state_dict
# state_dict 可以理解为是模型的基本信息（如：weight 和 bias）
print(model.state_dict()) # OrderedDict([('linear.weight', tensor([[-0.3402, -0.2288, -0.2258,  0.2858,  0.0122,  0.1359]])), ('linear.bias', tensor([0.1920]))])

torch.save(model.state_dict(), FILE)
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
  print(param)


# how to save checkpoint
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

checkpoint = {
  "epoch": 90,
  "model_state": model.state_dict(),
  "optim_state": optimizer.state_dict(),
}

torch.save(checkpoint, "checkpoint.pth")

loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

print(optimizer.state_dict()) # lr 更新为 0.01
