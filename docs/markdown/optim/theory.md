# Optimizer mathematics

Formulas implemented by `nn::optim`. How this library stores gradients and applies the update is in [mechanism.md](mechanism.md).

Notation for one parameter tensor, written elementwise:

- $\theta_t$: value after step $t$ ($\theta_0$ is the initial weight before the first step)
- $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$: gradient from backward (this library’s `param->grad()`)
- $\alpha$: learning rate
- $\lambda$: weight decay coefficient

All three classes use the **PyTorch-style** coefficient: $\lambda$ is applied to the parameter (or to the gradient) directly, not as $\frac{\lambda}{2}\|\theta\|^2$ inside a hand-written loss. `weight_decay = 0` makes the extra term a no-op; SGD/Adam still evaluate $g + \lambda\theta$ and AdamW still evaluates $\alpha\lambda\theta$. That is not inefficient: $\lambda$ is a loop-invariant scalar, so $g + \lambda\theta$ is one fused multiply-add per element (broadcast $\lambda$). When $\lambda = 0$ the hardware still does that FMA, but it is cheaper than a per-element branch (which also blocks SIMD) and cheaper than duplicating the whole update into two kernels. Next to Adam’s $\sqrt{\hat{v}_t}$, the extra FMA is noise.

---

## SGD

**Purpose.** Walk downhill on the loss: subtract a scaled gradient from the weights.

$$
\theta_t = \theta_{t-1} - \alpha\, (g_t + \lambda\, \theta_{t-1})
$$

When $\lambda = 0$ this is vanilla gradient descent:

$$
\theta_t = \theta_{t-1} - \alpha\, g_t
$$

**Weight decay as L2.** Adding $\lambda\theta$ to the gradient is the same gradient you would get from the regularized objective $\mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$ (the factor $\frac{1}{2}$ is absorbed into how $\lambda$ is defined). The extra term pulls weights toward zero and is the classic way to limit parameter's magnitude.

**Motivation.** One hyperparameter $\alpha$ and an optional $\lambda$. No per-parameter memory. Sensitive to gradient scale: a steep direction moves much farther than a flat one in a single step.

This library’s SGD has no momentum. Momentum would keep an exponential average of $g_t$ and step along that velocity; it is not implemented here.

---

## Adam

Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014). ([arxiv](https://arxiv.org/abs/1412.6980))

**Purpose.** Scale each coordinate of the update by a running estimate of gradient magnitude, and correct the bias of those estimates at the start of training.

Let $\beta_1, \beta_2 \in [0,1)$ (defaults $0.9$, $0.999$) and $\varepsilon > 0$ (default $10^{-8}$).

Coupled weight decay (same idea as SGD) is applied **before** the moments:

$$
g^{\mathrm{eff}}_t = g_t + \lambda\, \theta_{t-1}
$$

First moment (mean of the gradient):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g^{\mathrm{eff}}_t, \qquad m_0 = 0
$$

Second moment (uncentered variance):

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, (g^{\mathrm{eff}}_t)^2, \qquad v_0 = 0
$$

**Why two moments.** $m_t$ is a smoothed gradient: it damps noise and keeps a direction when a minibatch gradient flickers. $v_t$ tracks how large each coordinate has been. Dividing by $\sqrt{v_t}$ makes a parameter with huge gradients take smaller steps and a rarely updated parameter take relatively larger ones. That is the adaptive part of Adam.

**Bias correction.** With $m_0 = v_0 = 0$, the averages are biased toward zero for small $t$. Under a constant gradient the closed form is $m_t = (1-\beta_1^t)\, g$, so dividing by $1-\beta_1^t$ recovers $g$:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$\beta_2$ is close to 1, so $v_t$ stays underestimated longer than $m_t$; both corrections matter in the first few hundred steps.

**Parameter update.**

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

$\varepsilon$ avoids dividing by zero when a coordinate has never seen a nonzero gradient. It is added **outside** the square root (PyTorch / Kingma–Ba).

**Motivation.** Minibatch gradients are noisy and different layers have very different scales. Adam’s per-coordinate $\sqrt{v}$ denominator is a cheap diagonal preconditioner: you do not tune a separate learning rate per layer. The cost is two extra buffers the size of each parameter.

**Coupled decay.** Because $\lambda\theta$ is mixed into $g^{\mathrm{eff}}$, decay is scaled by the same $1/(\sqrt{\hat{v}}+\varepsilon)$ as the gradient. Large $|v|$ weakens both the loss gradient **and** the pull toward zero. That is the behaviour Loshchilov & Hutter argued against, and the reason AdamW exists.

---

## AdamW

Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2017). ([arxiv](https://arxiv.org/abs/1711.05101))

**Purpose.** Keep Adam’s adaptive steps on the **loss gradient only**, and apply weight decay as a separate shrink of $\theta$.

Moments use raw $g_t$, not $g_t + \lambda\theta$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

Update (decay uses the **pre-update** $\theta_{t-1}$):

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} - \alpha\lambda\, \theta_{t-1}
$$

which is the same as

$$
\theta_t = \theta_{t-1}\,(1 - \alpha\lambda) - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}.
$$

When $\lambda = 0$, this is identical to Adam.

**Why decoupling.** True weight decay is “multiply the weights by $1-\alpha\lambda$ each step.” Putting $\lambda\theta$ into Adam’s $g^{\mathrm{eff}}$ instead makes the decay **adaptive**: coordinates with large $v_t$ are decayed less. That is not the same regularizer as $\frac{\lambda}{2}\|\theta\|^2$ under SGD, and $\lambda$ then interacts with $\beta_2$ and $\varepsilon$. Decoupled decay keeps $\lambda$ on the same scale as in SGD: a property of the weights, not of the moment estimates.

**Default $\lambda$.** Adam defaults `weight_decay` to `0`. AdamW defaults it to `0.01`, matching the usual PyTorch `AdamW` default — decay is part of the method, not an afterthought.

---

## Side-by-side

| | SGD | Adam | AdamW |
| --- | --- | --- | --- |
| Direction | $g_t$ | bias-corrected $\hat{m}_t$ | same as Adam, from $g_t$ only |
| Scale | $\alpha$ | $\alpha / (\sqrt{\hat{v}_t}+\varepsilon)$ | same |
| $\lambda\theta$ goes into | the gradient | the gradient **and** therefore $m,v$ | the parameter, **after** $m,v$ |
| Extra state | none | $m,v,t$ per parameter | $m,v,t$ per parameter |

SGD with $\lambda>0$ and AdamW with the same $\lambda$ both shrink $\theta$ by a factor involving $\alpha\lambda$. Adam with $\lambda>0$ shrinks in gradient space and then adapts that shrink, so it will not match the other two even with identical $\alpha$ and $\lambda$.
