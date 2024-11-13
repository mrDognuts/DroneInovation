import torch
print(torch.__version__)  # Should print 2.5.1+cu121
print(torch.cuda.is_available())  # Should print True if GPU is enabled
print(torch.cuda.get_device_name(0))  # Should print the name of your GPU
