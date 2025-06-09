import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. 定义 PINN 网络 (Haiku MLP)
# ----------------------------------------
def mlp_fn(x_t):
    mlp = hk.Sequential([
        hk.Linear(64), jax.nn.tanh,
        hk.Linear(64), jax.nn.tanh,
        hk.Linear(64), jax.nn.tanh,
        hk.Linear(1),
    ])
    return mlp(x_t)

# pure-function transform
mlp = hk.without_apply_rng(hk.transform(mlp_fn))

# ----------------------------------------
# 2. PDE 残差 (无粘 Burgers)，对单点求标量导后 vmap 并行
# ----------------------------------------
def pde_residual(params, x, t):
    # x, t: shape [N,]
    # 定义对单个 (xi, ti) 返回标量 u 的模型
    def u_model(xi, ti):
        xt = jnp.stack([xi, ti], axis=-1)[None, :]   # [1,2]
        return mlp.apply(params, xt).squeeze()      # -> scalar

    # 构造针对标量输出的 grad
    grad_t = jax.grad(u_model, argnums=1)
    grad_x = jax.grad(u_model, argnums=0)

    # 并行化应用到整个批次
    u_t = jax.vmap(grad_t)(x, t)   # [N,]
    u_x = jax.vmap(grad_x)(x, t)   # [N,]

    # 批量预测 u 本身
    xt_batch = jnp.stack([x, t], axis=1)         # [N,2]
    u = mlp.apply(params, xt_batch).squeeze()   # [N,]

    # 返回残差向量
    return u_t + u * u_x                         # [N,]

# ----------------------------------------
# 3. 损失函数
# ----------------------------------------
def loss_fn(params, batch):
    x_int, t_int, x_ini, t_ini, x_bc, t_bc = batch
    # PDE 残差损失
    r = pde_residual(params, x_int, t_int)
    loss_pde = jnp.mean(r**2)
    # 初始条件损失: u(x,0)=1+sin(2pi x)
    u_ini_true = 1 + jnp.sin(2*jnp.pi*x_ini)
    u_ini_pred = mlp.apply(params, jnp.stack([x_ini, t_ini],1)).squeeze()
    loss_ini = jnp.mean((u_ini_pred - u_ini_true)**2)
    # 周期边界损失: u(0,t)=u(1,t)
    u0 = mlp.apply(params, jnp.stack([jnp.zeros_like(t_bc), t_bc],1)).squeeze()
    u1 = mlp.apply(params, jnp.stack([jnp.ones_like(t_bc),  t_bc],1)).squeeze()
    loss_bc = jnp.mean((u0 - u1)**2)
    return loss_pde + loss_ini + loss_bc

# ----------------------------------------
# 4. 训练步骤
# ----------------------------------------
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ----------------------------------------
# 5. 采样函数
# ----------------------------------------
def sampler(key, N_int, N_ini, N_bc, T):
    # 内部点 (uniform)
    key, sub = jax.random.split(key)
    x_int = jax.random.uniform(sub, (N_int,), minval=0.0, maxval=1.0)
    key, sub = jax.random.split(key)
    t_int = jax.random.uniform(sub, (N_int,), minval=0.0, maxval=T)
    # 初始条件点 t=0
    key, sub = jax.random.split(key)
    x_ini = jax.random.uniform(sub, (N_ini,), minval=0.0, maxval=1.0)
    t_ini = jnp.zeros_like(x_ini)
    # 边界点  x=0 和 x=1
    key, sub = jax.random.split(key)
    t_bc = jax.random.uniform(sub, (N_bc,), minval=0.0, maxval=T)
    x_bc = jnp.zeros_like(t_bc)   # 同时 x=1 在损失函数内部比较
    return (key, (x_int, t_int, x_ini, t_ini, x_bc, t_bc))

# ----------------------------------------
# 6. 主程序
# ----------------------------------------
def main():
    # 超参数
    N_int, N_ini, N_bc = 20000, 2000, 2000
    T = 0.3
    lr = 1e-3
    epochs = 25000

    # 初始化
    key = jax.random.PRNGKey(42)
    # 随机初始化网络参数
    dummy = jnp.zeros((1,2))
    params = mlp.init(key, dummy)
    global optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # 训练
    for ep in range(1, epochs+1):
        key, batch = sampler(key, N_int, N_ini, N_bc, T)
        params, opt_state, loss = train_step(params, opt_state, batch)
        if ep % 500 == 0:
            print(f"Epoch {ep}/{epochs}, Loss={loss:.3e}")

    # 评估和绘图
    nx, nt = 200, 3
    x_plot = np.linspace(0,1,nx)
    t_plot = np.array([0.0, T/2, T])

    plt.figure(figsize=(6,4))
    for ti in t_plot:
        xx = jnp.array(x_plot)
        tt = jnp.full_like(xx, ti)
        u_pred = mlp.apply(params, jnp.stack([xx, tt],1)).squeeze()
        plt.plot(x_plot, np.array(u_pred), label=f"t={ti:.2f}")

    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.title('PINN (JAX+Haiku+Optax) for Inviscid Burgers')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
