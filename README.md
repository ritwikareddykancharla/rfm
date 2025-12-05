# ⭐ Routing Foundation Model (RFM)
### *A Unified Neural Optimization Framework for Large-Scale Routing and MILPs*

<p align="left">
  <img src="https://img.shields.io/badge/status-in%20progress-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/pytorch-2.x-red?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-purple?style=flat-square" />
</p>

The **Routing Foundation Model (RFM)** is a neural surrogate optimization architecture designed to solve large-scale **routing** and **mixed-integer linear programs (MILPs)** at *neural network inference speed*.  
RFM builds on the insight that **transformers behave like unrolled optimizers**, enabling them to approximate MILP reasoning when equipped with the right inductive biases.

RFM combines:
- 🧠 **MILP-aware attention (Aᵀv dual correction)**
- 🔧 **constraint-specialized Mixture-of-Experts**
- 🔄 **latent gradient refinement**
- 🧮 **soft integer relaxations**
- ✨ **iterative transformer-like updates**

RFM is intended for Amazon-scale middle-mile routing, VRP variants, supply chain optimization, and general combinatorial optimization tasks.

---

## 🚀 Features

### 🔷 Transformer as an Optimizer  
Attention ≈ proximal update, residuals ≈ gradient descent, dual terms ≈ feasibility correction.

### 🔷 MILP-Aware Attention  
Injects feasibility structure directly into logits:
```
L = QKᵀ/√d - γ * (Aᵀv)
```

### 🔷 Mixture-of-Experts for Constraints  
Experts specialize to:
- flow conservation  
- capacity  
- activation/binary coupling  
- SLA/time-window constraints  

### 🔷 Latent Optimization Loop  
Gradient-like refinement mimics interior-point / dual ascent behavior.

### 🔷 Extensible  
Replace encoder with GNN/Graphormer/Mamba, add diffusion priors, or warm-start Gurobi.

---

## 🏗️ Repository Structure

```
rfm/
│
├── rfm/
│   ├── models/
│   │   ├── milp_transformer.py
│   │   └── embeddings.py
│   │
│   ├── layers/
│   │   ├── milp_attention.py
│   │   └── feedforward.py
│   │
│   ├── solvers/
│   │   ├── constraint_experts.py
│   │   └── refinement.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   └── dataset.py
│   │
│   ├── utils/
│   │   ├── milp.py
│   │   └── graph.py
│   │
│   └── __init__.py
│
├── experiments/
│   └── synthetic_50_nodes.py
│
├── paper/
│   ├── RFM_monograph.tex
│   └── references.bib
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### 1️⃣ Clone the repo
```bash
git clone https://github.com/ritwikareddykancharla/rfm.git
cd rfm
```

### 2️⃣ Install dependencies
```bash
pip install -e .
```

### 3️⃣ Run a synthetic routing problem
```bash
python experiments/synthetic_50_nodes.py
```

---

## 🔬 How RFM Works

### 1. Soft binary relaxation
```python
x = sigmoid(logits / tau)
```

### 2. Constraint violations  
```python
v = relu(A @ x - b)
```

### 3. Dual-inspired correction  
```python
h = A.T @ v
```

### 4. MILP-aware attention  
```python
L = QK^T / sqrt(d) - γ * h
α = softmax(L)
```

### 5. Constraint experts refine feasibility  
```python
Φ = MoE(v)
```

### 6. Latent gradient-style refinement  
```python
x = x - η * ∇(cᵀx + Φ)
```

---

## 📚 Citation

```bibtex
@misc{kancharla2025rfm,
  title={Routing Foundation Model (RFM): A Unified Neural Optimization Framework for Large-Scale Routing and MILPs},
  author={Kancharla, Ritwika},
  year={2025},
  archivePrefix={arXiv},
}
```

---

## 📬 Contact  
**Ritwika Kancharla**  
📧 ritwikareddykancharla@gmail.com
