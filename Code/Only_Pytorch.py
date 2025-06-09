import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------
# 1. 定义一个全连接网络（MLP）
# ----------------------------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers)-1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            # 最后一个线性层后不加激活
            if i < len(layers)-2:
                layer_list.append(self.activation)
        self.net = nn.Sequential(*layer_list)

    def forward(self, x, t):
        # 网络输入是二维坐标 (x,t)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# ----------------------------------------
# 2. 采样
# ----------------------------------------
def sampler(N_interior, N_initial):
    # PDE 内部点
    x_int = torch.rand(N_interior,1, requires_grad=True)
    t_int = torch.rand(N_interior,1, requires_grad=True) * T

    # 初始条件点 (t=0)
    x_init = torch.rand(N_initial,1)
    t_init = torch.zeros_like(x_init)
    u_init = 1.0 + torch.sin(2*torch.pi*x_init)

    return x_int, t_int, x_init, t_init, u_init

# ----------------------------------------
# 3. PDE 残差计算
# ----------------------------------------
def pde_residual(model, x, t):
    u = model(x, t)
    # 自动求导 u_t
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    # 自动求导 u_x
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    # Burgers PDE: u_t + u * u_x = 0
    return u_t + u * u_x

# ----------------------------------------
# 4. 训练设置
# ----------------------------------------
# 超参数
layers = [2, 64, 64, 64, 1]   # 2→64→64→64→1
N_int     = 20000            # 内部采样点
N_ini     = 2000             # 初始条件点
T         = 0.3              # 时间上界
lr        = 1e-3             # 学习率
epochs    = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = PINN(layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# ----------------------------------------
# 5. 训练循环
# ----------------------------------------
for ep in range(1, epochs+1):
    # 采样
    x_int, t_int, x_ini, t_ini, u_ini = sampler(N_int, N_ini)
    x_int, t_int = x_int.to(device), t_int.to(device)
    x_ini, t_ini, u_ini = x_ini.to(device), t_ini.to(device), u_ini.to(device)

    # PDE 残差损失
    res = pde_residual(model, x_int, t_int)
    loss_pde = torch.mean(res**2)

    # 初始条件损失
    u_pred_ini = model(x_ini, t_ini)
    loss_ini = torch.mean((u_pred_ini - u_ini)**2)

    # 周期边界损失 u(0,t)=u(1,t)
    t_b = torch.rand(N_ini,1).to(device) * T
    u0 = model(torch.zeros_like(t_b), t_b)
    u1 = model(torch.ones_like(t_b),  t_b)
    loss_bc = torch.mean((u0-u1)**2)

    # 合并总损失
    loss = loss_pde + loss_ini + loss_bc

    # 梯度下降
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每 500 轮打印一次
    if ep % 500 == 0:
        print(f"Epoch {ep}/{epochs} — Loss PDE: {loss_pde.item():.3e}, "
              f"Ini: {loss_ini.item():.3e}, BC: {loss_bc.item():.3e}")

# ----------------------------------------
# 6. 结果可视化
# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 网格评估
nx, nt = 200, 200
x = torch.linspace(0,1,nx).unsqueeze(1).to(device)
t = torch.linspace(0,T,nt).unsqueeze(1).to(device)
X, Tt = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
x_flat = X.reshape(-1,1)
t_flat = Tt.reshape(-1,1)
u_pred = model(x_flat, t_flat).detach().cpu().numpy().reshape(nx,nt)

# 绘制 t=0、t=T/2、t=T 三个时刻的解
plt.figure(figsize=(8,5))
for ti in [0, nt//2, nt-1]:
    plt.plot(x.cpu().numpy(), u_pred[:,ti], label=f"t={t.cpu().numpy()[ti].item():.2f}")
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('PINN 求解 Inviscid Burgers 方程')
plt.grid(True)
plt.show()