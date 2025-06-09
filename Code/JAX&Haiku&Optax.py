import jax, jax.numpy as jnp
import haiku as hk
import optax

# 1. 网络定义
def mlp_fn(x_t):
    # x_t: [batch,2]，列是 x,t
    mlp = hk.Sequential([
        hk.Linear(64), jax.nn.tanh,
        hk.Linear(64), jax.nn.tanh,
        hk.Linear(64), jax.nn.tanh,
        hk.Linear(1),
    ])
    return mlp(x_t)

# 将它 transform 成 pure-function + param/state
mlp = hk.without_apply_rng(hk.transform(mlp_fn))

# 2. PDE 残差
def residual(params, x, t):
    x_t = jnp.stack([x, t], axis=1)      # [N,2]
    u = mlp.apply(params, x_t).squeeze() # [N]
    u_t = jax.grad(lambda tt: mlp.apply(params, jnp.stack([x, tt],1)).squeeze())(t)
    u_x = jax.grad(lambda xx: mlp.apply(params, jnp.stack([xx, t],1)).squeeze())(x)
    return u_t + u * u_x                  # [N]

# 3. 损失函数
def loss_fn(params, batch):
    x_int, t_int, x_ini, t_ini = batch
    # PDE 残差损失
    r = residual(params, x_int, t_int)
    loss_pde = jnp.mean(r**2)
    # 初始条件损失
    u_ini = 1 + jnp.sin(2*jnp.pi*x_ini)
    pred_ini = mlp.apply(params, jnp.stack([x_ini, t_ini],1)).squeeze()
    loss_ini = jnp.mean((pred_ini - u_ini)**2)
    # 周期边界损失
    t_b = t_ini
    u0 = mlp.apply(params, jnp.stack([jnp.zeros_like(t_b), t_b],1)).squeeze()
    u1 = mlp.apply(params, jnp.stack([jnp.ones_like(t_b), t_b],1)).squeeze()
    loss_bc = jnp.mean((u0 - u1)**2)
    return loss_pde + loss_ini + loss_bc

# 4. 优化器
opt = optax.adam(1e-3)

# 5. 初始化参数
key = jax.random.PRNGKey(42)
# 用一些点来初始化网络 shape
x0 = jnp.zeros((1,)); t0 = jnp.zeros((1,))
params = mlp.init(key, jnp.stack([x0, t0],1))

opt_state = opt.init(params)

# 6. 训练一步函数 jit 加速
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 然后在一个循环里不断调用 train_step，传入采样好的 batch