import numpy as np
import torch
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ref = cv2.imread('test_img/ref.tif', cv2.IMREAD_GRAYSCALE)
cap = cv2.imread('test_img/cap.tif', cv2.IMREAD_GRAYSCALE)
ref = ref.astype(np.float64) / 255.0
cap = cap.astype(np.float64) / 255.0
# 初始化变量
ref = torch.tensor(ref, dtype=torch.float64).to(device)
cap = torch.tensor(cap, dtype=torch.float64).to(device)
cons = cap - ref
pha = torch.zeros_like(ref, device=device, dtype=torch.float64, requires_grad=True)
N, M = ref.shape
h = 1
optimizer = optim.Adam([pha], lr=0.1)

def compute_loss(pha, ref, cons):
    # 计算梯度
    pha_grad_x = torch.cat((pha[:, 1:] - pha[:, :-1], torch.zeros(N, 1, device=device)), 1)
    pha_grad_y = torch.cat((pha[1:, :] - pha[:-1, :], torch.zeros(1, M, device=device)), 0)

    ref_grad_x = torch.cat((ref[:, 1:] - ref[:, :-1], torch.zeros(N, 1, device=device)), 1)
    ref_grad_y = torch.cat((ref[1:, :] - ref[:-1, :], torch.zeros(1, M, device=device)), 0)

    # 计算点积和添加cons
    grad_dot = pha_grad_x * ref_grad_x + pha_grad_y * ref_grad_y + cons

    # 使用均方误差作为损失函数
    loss = (grad_dot ** 2).mean()
    return loss


# 运行迭代
iterations = 5000
for it in tqdm(range(iterations)):
    optimizer.zero_grad()
    loss = compute_loss(pha, ref, cons)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pha %= (2 * torch.pi)

    if it % 100 == 0:
        print(f"Iteration {it}: Loss {loss.item()}")

# 检查结果
pha_cpu = pha.detach().cpu().numpy()
plt.imshow(pha_cpu)
plt.show()
print("Optimization finished. Loss:", loss.item())

# # 迭代参数
# max_iter = 10
# tolerance = 1e-6
# omega = 1.5  # 松弛因子，用于加速收敛
#
# # 迭代求解
# for iteration in tqdm(range(max_iter)):
#     pha_old = pha.copy()
#
#     for i in range(1, height + 1):
#         for j in range(1, width + 1):
#             # 使用中心差分公式计算梯度
#             grad_pha_x = (pha_old[i, j + 1] - pha_old[i, j - 1]) / (2 * h)
#             grad_pha_y = (pha_old[i + 1, j] - pha_old[i - 1, j]) / (2 * h)
#             grad_ref_x = (ref[i - 1, j - 1] - ref[i - 1, j - 1]) / (2 * h)  # 注意边界匹配
#             grad_ref_y = (ref[i - 1, j - 1] - ref[i - 1, j - 1]) / (2 * h)
#
#             # 计算在(i, j)处的差分方程值
#             pha[i, j] = (omega / 4.0) * ((pha_old[i, j + 1] + pha[i, j - 1] + pha_old[i + 1, j] + pha[i - 1, j]) -
#                                          h ** 2 * cons[i - 1, j - 1] / (grad_ref_x ** 2 + grad_ref_y ** 2 + 1e-10)) \
#                         + (1 - omega) * pha_old[i, j]
#
#     # 计算误差，检查是否达到收敛标准
#     error = np.max(np.abs(pha - pha_old))
#     print(error)
#     if error < tolerance:
#         print("Converged after", iteration, "iterations")
#         break
# else:
#     print("Max iterations reached without convergence.")
#
# # 去掉padding
# pha = pha[1:-1, 1:-1]
#
# # 绘制结果
# plt.imshow(pha, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Computed Pha')
# plt.show()




# gt = torch.zeros_like(ref, device=device, dtype=torch.float64, requires_grad=True)
# a = torch.nn.Parameter(torch.ones(1).to(device), requires_grad=True)
# # 优化器
# loss_fn = torch.nn.MSELoss()

# writer = SummaryWriter('runs/pha_optimization')
# # 迭代优化
# iterations = 3000
# for it in range(iterations):
#     optimizer.zero_grad()
#
#     # 计算梯度
#     grad_pha_x, grad_pha_y = torch.gradient(pha)
#     grad_ref_x, grad_ref_y = torch.gradient(ref)
#
#     # 计算 ∇pha ⋅ ∇ref + cons = 0
#
#     loss1 = torch.mean((grad_pha_x * grad_ref_x + grad_pha_y * grad_ref_y + cons))
#
#     loss = loss_fn(loss1, gt)
#     # 反向传播
#     loss.backward()
#
#     # 更新pha
#     optimizer.step()
#
#     if it % 10 == 0:
#         print(f"Iteration {it}: Loss {loss.item()}")
#         writer.add_scalar('Loss', loss.item(), it)
#         # 将pha矩阵调整为[1, 1000, 1000]的形式，适合add_image
#         pha_img = pha.unsqueeze(0)  # [1, 1000, 1000]
#         pha_img = (pha_img - pha_img.min()) / (pha_img.max() - pha_img.min())  # 归一化到[0,1]
#         writer.add_image('Pha Matrix', pha_img, it)
# writer.close()

print("Optimization Finished.")
