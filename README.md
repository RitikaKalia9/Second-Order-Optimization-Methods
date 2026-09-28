# Second-Order Optimization Methods in PyTorch

A hands-on PyTorch lab exploring **curvature, Newton's method, Hessian computation, quasi-Newton optimization, and Hessian-free methods**.

The demonstration of why second-order optimization can converge in very few iterations, while also showing why explicitly computing and storing the Hessian becomes impractical for large neural networks.

## Overview

**Second-Order Optimization Methods**.

It covers:

- Gradient and Hessian computation using PyTorch autograd
- Newton's method and Newton steps
- Newton vs. gradient descent on a stretched quadratic valley
- The computational and memory cost of a full Hessian
- Gauss–Newton optimization
- Levenberg–Marquardt damping
- L-BFGS quasi-Newton optimization
- Hessian-free optimization using Hessian-vector products
- Conjugate gradient for solving Newton systems
- A practical comparison of first-order and second-order methods

## Learning Objectives

After completing this lab, the following concepts can be demonstrated:

1. Compute gradients and Hessians using PyTorch autograd.
2. Explain the role of `create_graph=True` when calculating second derivatives.
3. Apply Newton's method to quadratic objectives.
4. Understand how curvature information affects optimization on poorly conditioned problems.
5. Measure the memory growth of a full Hessian.
6. Explain why explicitly forming a Hessian is impractical for large models.
7. Implement Gauss–Newton and Levenberg–Marquardt updates.
8. Use L-BFGS through `torch.optim`.
9. Compute Hessian-vector products without explicitly constructing the Hessian.
10. Use conjugate gradient to obtain an approximate Newton direction.

## Requirements

- Python 3.9+
- PyTorch 2.x
- CPU is sufficient
- Jupyter Notebook / JupyterLab recommended

Install PyTorch if required:

```bash
pip install torch
```

No GPU is required for the experiments in this lab.

## Getting Started

Clone or download the repository and open the notebook containing the lab exercises.

Example:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Then launch Jupyter:

```bash
jupyter notebook
```

Run the notebook from top to bottom using a fresh kernel.

## Lab Structure

### 1. Gradient, Hessian and Newton Step

The first experiment uses:

```text
L(w) = w² - 4w + 5
```

starting from:

```text
w₀ = 5
```

Autograd is used to calculate the gradient and Hessian.

Expected values:

```text
g = 6
H = 2
w₁ = 2
```

The loss changes from:

```text
L(w₀) = 10
```

to:

```text
L(w₁) = 1
```

Because the objective is quadratic, its second-order Taylor expansion exactly represents the function. Therefore, Newton's method reaches the minimum in one step.

A key PyTorch concept is:

```python
g = torch.autograd.grad(L(w), w, create_graph=True)[0]
H = torch.autograd.grad(g, w)[0]
```

`create_graph=True` keeps the gradient differentiable so that a second derivative can be computed.

---

### 2. Newton vs. Gradient Descent

A stretched quadratic bowl is used to show the effect of poor conditioning.

The objective uses:

```text
A = [[20, 0],
     [0, 1]]
```

The optimum is:

```text
[0.1, 1.0]
```

Gradient descent requires many iterations because the steep direction restricts the stable learning rate.

The experiment uses a learning rate of `0.09`.

After 60 gradient-descent iterations:

```text
error ≈ 0.017436
```

Newton's method reaches the optimum in one step:

```text
Newton 1 step error ≈ 9.69 × 10⁻⁸
```

The experiment illustrates that curvature information determines both the direction and effective step length of the Newton update.

---

### 3. Cost of the Full Hessian

The third experiment measures how the Hessian scales with the number of model parameters.

For a model with `n` parameters:

```text
Hessian storage = O(n²)
```

The lab measures Hessian construction time and memory for several network sizes.

Example results:

| Parameters | Hessian Time | Hessian Memory |
|---:|---:|---:|
| 49 | 0.012 s | 0.01 MB |
| 193 | 0.048 s | 0.14 MB |
| 769 | 0.218 s | 2.26 MB |
| 3073 | 1.059 s | 36.02 MB |

The experiment demonstrates the quadratic memory growth.

For a model with:

```text
n = 10,000,000 parameters
```

the estimated Hessian storage is approximately:

```text
364 TB
```

This makes explicitly forming and storing a full Hessian impractical for modern deep-learning models.

Newton's method also requires solving a linear system involving the Hessian, which can introduce approximately `O(n³)` computational cost for a direct solve.

---

### 4. Gauss–Newton and Levenberg–Marquardt

The fourth experiment considers a nonlinear least-squares problem.

The model fits:

```text
θ₀ exp(θ₁x)
```

to noisy samples generated from:

```text
3 exp(-1.5x)
```

The Gauss–Newton approximation uses the Jacobian of the residuals instead of the full Hessian.

Levenberg–Marquardt adds damping:

```text
JᵀJ + λI
```

This improves numerical stability when `JᵀJ` is nearly singular.

The experiment compares:

- Gauss–Newton: `λ = 0`
- Mild damping: `λ = 1`
- Heavy damping: `λ = 1000`

Expected behavior:

- Undamped Gauss–Newton converges quickly.
- Mild damping slows the early progress but reaches the same solution.
- Heavy damping behaves more like gradient descent and progresses much more slowly.

The practical LM strategy is to decrease `λ` after successful steps and increase it when a step fails.

---

### 5. L-BFGS

L-BFGS is a **quasi-Newton method**.

Unlike classical Newton's method, L-BFGS does not explicitly construct the full Hessian.

Instead, it builds an implicit approximation to the inverse Hessian from recent gradient differences.

The lab compares L-BFGS with Adam.

Expected results:

```text
Adam 10 steps  → SSE = 9.88366
Adam 50 steps  → SSE = 0.07337
Adam 200 steps → SSE = 0.01275
Adam 500 steps → SSE = 0.01275

L-BFGS 1 step → SSE = 0.01275
L-BFGS 2 steps → SSE = 0.01275
L-BFGS 5 steps → SSE = 0.01275
```

Important interpretation:

A single `optimizer.step()` for L-BFGS can perform multiple internal iterations because `max_iter=20` allows the optimizer to evaluate the closure repeatedly.

Therefore, the comparison should not be interpreted as L-BFGS being twenty times cheaper.

L-BFGS is also most suitable for deterministic, full-batch objectives. Changing mini-batches can make its curvature estimates inconsistent.

---

### 6. Hessian-Free Optimization

The final optimization method avoids constructing the Hessian completely.

A Hessian-vector product is computed using:

```python
def hvp(f, w, v):
    g = torch.autograd.grad(f(w), w, create_graph=True)[0]
    return torch.autograd.grad(g @ v, w)[0]
```

This computes:

```text
H v
```

without explicitly creating the matrix `H`.

The memory requirement therefore changes from approximately:

```text
O(n²)
```

for a full Hessian to:

```text
O(n)
```

for Hessian-vector based computation.

Conjugate gradient is then used to solve:

```text
HΔw = -g
```

using only Hessian-vector products.

For the quadratic example, the computed direction is:

```text
[-2.9, 5.0]
```

which matches the exact Newton direction and takes the parameters to:

```text
[0.1, 1.0]
```

without explicitly constructing the Hessian.

---

## Methods Compared

| Method | Curvature Information | Memory | Typical Use |
|---|---|---:|---|
| Gradient Descent | None | O(n) | Large deep neural networks |
| Newton | Exact Hessian | O(n²) | Small models and teaching |
| Gauss–Newton / LM | Jacobian-based approximation | O(n²) or less | Least-squares and curve fitting |
| L-BFGS | Implicit from past gradients | O(mn) | Full-batch medium-sized problems |
| Hessian-Free | Hessian-vector products | O(n) | Large scientific/research models |

## Key Takeaways

### Why use second-order optimization?

First-order methods use only the gradient:

```text
g = ∇L
```

Second-order methods also use curvature information:

```text
H = ∇²L
```

The additional curvature information can produce much better update directions and significantly reduce the number of iterations, especially for poorly conditioned problems.

However, the full Hessian has quadratic memory requirements:

```text
O(n²)
```

and direct methods for solving Hessian systems can become very expensive.

Therefore, practical optimization methods avoid the full Hessian when models become large.

### Practical alternatives

The lab demonstrates three important alternatives:

**Gauss–Newton / Levenberg–Marquardt**

Uses a Jacobian-based curvature approximation and damping.

**L-BFGS**

Stores a limited history of gradient information instead of the full Hessian.

**Hessian-Free Optimization**

Uses Hessian-vector products and conjugate gradient without ever forming the Hessian matrix.

## Main PyTorch Concepts

The lab makes use of:

```python
torch.autograd.grad()
torch.autograd.functional.hessian()
torch.autograd.functional.jacobian()
torch.linalg.solve()
torch.optim.Adam
torch.optim.LBFGS
torch.func.functional_call()
```

A particularly important pattern is:

```python
create_graph=True
```

whenever a gradient must be differentiated again.

## Lab Exercises

The tutorial includes six exercises:

1. Apply Newton's method to `3w² - 12w + 7`.
2. Change the quadratic matrix and relate gradient-descent iterations to the condition number.
3. Apply Newton's method to a non-convex polynomial and analyze the effect of the Hessian sign.
4. Implement adaptive Levenberg–Marquardt damping.
5. Compare L-BFGS and Adam on a small MLP using full-batch and mini-batch data.
6. Study truncated conjugate-gradient iterations on a 20-dimensional random quadratic.

## Submission Checklist

Before submission:

- Run every notebook cell from top to bottom using a fresh kernel.
- Ensure `create_graph=True` is present wherever a second derivative is required.
- Verify that outputs are reproducible.
- Answer all six lab exercises.
- Support each written answer with the relevant numerical output.
- Submit the completed notebook together with the written answers.

## References

This repository is based on the provided **Lab Tutorial 7 — Second-Order Optimization Methods** material for the Neural Networks and Deep Learning course.

Topics covered include:

- Newton's method
- Hessian-based optimization
- Gauss–Newton
- Levenberg–Marquardt
- L-BFGS
- Hessian-vector products
- Conjugate gradient
- Practical trade-offs between curvature accuracy and computational cost
