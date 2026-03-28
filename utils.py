import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

def save_config(opt, file_path):
    with open(file_path, "w") as f:
        json.dump(opt.__dict__, f, indent=2)

def cosine_similarity_matrix(a,b):
    if 'numpy' in str(type(a)):
        return cosine_similarity(a,b)
    else:
        return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)

# import numpy as np
# a = np.array([[1,2,0,1],[1,2,0,3]])

# print(a.argmax(axis=0).tolist())

# print((-a).argsort().tolist())

# sims = torch.Tensor([[1,1,0],[3,1,0],[1,2,3]]).cuda()
# # print(a[2:,2:])
# selelct_ = (sims.diag()>1).data.cpu().numpy().tolist()
# mask_diag = torch.zeros_like(sims.diag()).to(sims.device)
# for i in selelct_:
#     if i:
#         mask_diag[i] = 1
# print(mask_diag)
# print(a.diag())
# max_ = a.max(dim=1)   
# valuses, indices=max_[0].detach().cpu().numpy().tolist(), max_[1].detach().cpu().numpy().tolist()
# print(valuses, indices)
# a = torch.Tensor([0,1,2,3,4])
# print((a>2))
# (a>2).data.cpu().numpy().tolist()

# sims_matrix = torch.zeros((29000,1)).cuda()+2
# sims_matrix1 = torch.zeros((29000,1)).cuda()+1
# print( torch.gt(sims_matrix,sims_matrix1)+0)
sims = torch.Tensor([[1,2],[3,4],[5,6]]).cuda()
sims[torch.Tensor([0,2]).long(),0] = torch.Tensor([10,10]).cuda()

# print(sims)



