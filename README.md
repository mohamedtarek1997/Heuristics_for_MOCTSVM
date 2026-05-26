# MOCTSVM: Multi-Objective Optimal Classification Trees with SVM Splits

## Overview

MOCTSVM is a novel approach for learning optimal decision trees that use Support Vector Machines (SVM) as splitting criteria at internal nodes. Instead of axis-aligned splits (like CART), MOCTSVM uses **full SVM hyperplanes** as splitting rules, enabling richer, diagonal decision boundaries.

## Core Idea

Traditional decision trees split on a single feature (`feature_i > threshold`). MOCTSVM splits using an **SVM hyperplane** at each node: `sign(ω·x + ω₀)`. This allows each split to consider all features simultaneously, creating more expressive and potentially more accurate trees.

## Why MOCTSVM?

| Feature | CART | MOCTSVM |
|---------|------|---------|
| Split type | Axis-aligned | Oblique (hyperplane) |
| Features per split | 1 | All features |
| Margin optimization | No | Yes (via SVM) |
| Multi-objective | No | Yes (4 objectives) |

## The Optimization Problem

Four competing objectives are optimized simultaneously:

1. **Margin Maximization** - Maximize SVM margin at each split (better generalization)
2. **Misclassification Error** - Minimize wrong predictions at leaves
3. **SVM Hinge Loss** - Penalize margin violations
4. **Tree Complexity** - Penalize too many splits (regularization)

## Two Optimization Strategies

### 1. Genetic Algorithm (NSGA-II)
- Multi-objective optimization
- Returns Pareto front of trade-off solutions
- Best for understanding objective relationships

### 2. Simulated Annealing  
- Single-objective (weighted sum)
- Faster convergence
- Best for practical deployment

## Installation

```bash
pip install numpy pandas scikit-learn pymoo matplotlib
