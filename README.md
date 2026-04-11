# MPI Programming with Python

This repository contains assignment programs developed for the course:

**Advanced Architecture & High Performance Computing (PGCMSCC 2.2)**  
Master’s Program in Computer Science  
Ramakrishna Mission Vivekananda Centenary College, Rahara

---

## 📌 Overview

This repository demonstrates the implementation of parallel algorithms using **MPI (Message Passing Interface)** in Python (`mpi4py`). The focus is on applying **divide-and-conquer strategies at the hardware level** to efficiently solve computational problems.

The assignments emphasize performance-oriented computing and parallel execution across multiple processes.

---

## 🧠 Learning Outcomes

- Understanding MPI-based parallel programming
- Applying divide-and-conquer techniques in distributed systems
- Implementing computationally intensive algorithms efficiently
- Gaining hands-on experience with high-performance computing concepts

---

## 📂 Program Categories

### 1. 🔢 Linear Algebra
- Matrix Rank (Gaussian Elimination)
- Eigenvalues (Power Iteration)
- Eigenvectors (Power Iteration)
- Matrix Inverse (Gauss-Jordan Method)

---

### 2. 📈 Interpolation Algorithms
- Linear Interpolation
- Cubic Interpolation (Lagrange Form)
- Cosine Interpolation
- Nearest Neighbor Interpolation

---

### 3. 🤖 Machine Learning Algorithms
- KNN Algorithm Implementation 
- K-Means Clustering Algorithm Implementation 

---

### 4. ⇄ Open MP (C Programs)
- Bubble Sort using Open MP
- Matrix Addition using Open MP
- Heat Equation Simulation using Open MP and MPI

---

## ⚙️ Technologies Used

- Python
- mpi4py
- NumPy
- MPI (Message Passing Interface)
- Open MP

---

## 🚀 How to Run

Make sure MPI is installed (e.g., OpenMPI or MPICH), then run:

```bash
mpirun -np <number_of_processes> python <filename.py>
```


Repo Structure
```bash
├── linear_algebra/
├── interpolation/
├── machine_learning/
├── open_mp/
├── README.md
```


Maintained By Nishanka Das, M.Sc. CS Student @ RKMVCC, Rahara. Session 2025-26
