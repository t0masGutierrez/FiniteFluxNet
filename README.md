# Problem

The governing conservation law is

$$
\frac{\partial u}{\partial t}
+
\frac{\partial f(u)}{\partial x}
=0.
$$

$$
\frac{d}{dt}(\text{cell length}) = \text{inflow} - \text{outflow}
$$

with periodic boundary conditions

$$
u(t,0)=u(t,1).
$$

The initial conditions are sampled from Gaussian process

$$
u(0,\cdot)\sim\mathcal{N}(0,k).
$$

The covariance kernel is

$$
k(x,x') =
\sigma^2
\exp\biggl(
-\frac{1-\cos\!\bigl(2\pi(x-x'))}
{\ell^2}
\biggr).
$$

### Notation

| Symbol | Meaning |
|--------|---------|
| $u$ | Conserved quantity |
| $x$ | Spatial coordinate |
| $t$ | Time |
| $f$ | Flux function |
| $k$ | Covariance kernel |
| $\sigma$ | Standard deviation |
| $\ell$ | Correlation length |

---

# Neural Network

The model predicts the future state of a query trajectory using a set of context trajectories.

$$
\mathcal{T}_{\theta}:
\bigl(
u_q(t,x),
\lbrace u_i(t,x)\,u_i(t+\tau,x)\rbrace_{i=1}^{k}
\bigr)
\longmapsto
\hat{u}_q(t+\tau,x).
$$

### Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{T}_{\theta}$ | Neural network with parameters $\theta$ |
| $u_q(t,x)$ | Query input state |
| $\hat{u}_q(t+\tau,x)$ | Predicted future query state |
| $u_i(t,x)$ | Context input state |
| $u_i(t+\tau,x)$ | Context target state |
| $k$ | Number of context examples |
| $\tau$ | Time step |

---

# In-Distribution Flux Functions

$$
f(u)=au^3+bu^2+cu,
$$

where

$$
a,b,c\sim U([-1,1]).
$$

# Forward and Reverse Operators

$$
F_{f,\tau}:
u(t,x)
\longmapsto
u(t+\tau,x).
$$

$$
R_{f,\tau}:
u(t+\tau,x)
\longmapsto
\lbrace v \mid F_{f,\tau}(v)=u(t+\tau,x)\rbrace.
$$

### Notation

| Symbol | Meaning |
|--------|---------|
| $F_{f,\tau}$ | Forward evolution operator |
| $R_{f,\tau}$ | Reverse operator |
| $v$ | Candidate previous state |

---

# Out-of-Distribution Flux Functions

$$
f(u)=\sin(u)-\cos(u)
$$

and

$$
f(u)=\tanh(u).
$$
