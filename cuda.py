# import torch
#
# if torch.cuda.is_available():
#     print("CUDA is available.")
# else:
#     print("CUDA is not available.")
# import torch
#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("GPU Name:", torch.cuda.get_device_name(device))
# else:
#     print("CUDA is not available.")
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print("GPU Properties:", properties)
else:
    print("CUDA is not available.")
