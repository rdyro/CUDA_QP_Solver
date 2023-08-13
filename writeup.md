# Non-allocating Quadratic Program Solver on the GPU

---

## Introduction

---

## Quadratic Programming (QP) via Alternating Direction Method of Multipliers (ADMM)

### QP Problem Definition

### The Model Predictive Control (MPC) Test Problem

---

## Implementation of Necessary Tools

### Memory Allocation Routine

### Vector Utilities

### Sparse Matrix Utilities

### Linear System Solver

### Alternating Direction Method of Multipliers (ADMM) Loop

---

## Kernel Optimization Improvements

TODO: show how to do timing

### Disabling Dynamic Bounds Checking and Loop Reordering

### Faster Memory - Shared Memory

### Approximate Minimum Degree (AMD) Reordering

---

## CPU vs GPU Performance

---

# References

- [https://irhum.github.io/blog/cudajulia/](https://irhum.github.io/blog/cudajulia/)
- 