import jax
import jax.numpy as jnp
from jax import jit

# ----------------------------------------
# 1. Lax–Friedrichs 时间步更新
# ----------------------------------------
@jit
def lax_friedrichs_step(u, dt, dx):
    # 通量 f(u) = u^2 / 2
    f = 0.5 * u**2

    # 周期边界：使用 jnp.roll 实现 u_{j±1}
    up = jnp.roll(u, -1)
    um = jnp.roll(u, +1)
    fp = jnp.roll(f, -1)
    fm = jnp.roll(f, +1)

    # Lax–Friedrichs 更新公式
    return 0.5 * (up + um) - dt / (2 * dx) * (fp - fm)

# ----------------------------------------
# 2. 主求解函数
# ----------------------------------------
def solve_burgers_jax(Nx=400, T=0.3, CFL=0.4):
    dx = 1.0 / Nx
    # 空间网格：0, dx, 2dx, ..., 1-dx
    x = jnp.linspace(0.0, 1.0 - dx, Nx)

    # 初始条件
    u = 1.0 + jnp.sin(2 * jnp.pi * x)
    t = 0.0

    # 时间推进
    def cond(carry):
        u, t = carry
        return t < T

    def body(carry):
        u, t = carry
        umax = jnp.max(jnp.abs(u))
        # 自适应 dt
        dt = CFL * dx / jnp.where(umax > 1e-6, umax, 1e-6)
        dt = jnp.minimum(dt, T - t)
        u_new = lax_friedrichs_step(u, dt, dx)
        return (u_new, t + dt)

    # 用 while_loop 迭代到 t >= T
    u_final, _ = jax.lax.while_loop(cond, body, (u, t))
    return x, u_final

# ----------------------------------------
# 3. 可视化示例
# ----------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 参数
    Nx, T, CFL = 400, 0.3, 0.4
    x, u = solve_burgers_jax(Nx, T, CFL)

    plt.figure(figsize=(6,4))
    plt.plot(x, u, '-k', lw=1.5)
    plt.xlabel('x')
    plt.ylabel(f'u(x, t={T:.2f})')
    plt.title('Inviscid Burgers via Lax–Friedrichs (JAX)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()