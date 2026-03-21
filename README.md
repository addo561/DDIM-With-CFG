# DDIM with CFG, with Medical Mnist

Trained a diffusion model with DDPM, sampled with DDIM for fast deterministic 
generation, then extended with Classifier-Free Guidance (CFG) for conditional control.

---

## Theory

### DDIM Sampling

DDIM (Song et al., 2020) replaces the stochastic DDPM reverse step with a 
deterministic update, allowing generation in far fewer steps (~50 vs 1000).
<img width="876" height="178" alt="Screenshot 2026-03-08 at 11 36 29 PM" src="https://github.com/user-attachments/assets/7193b149-f163-4c38-a71e-85983d28321b" />

### DDIM Samples(10x faster)

<img width="515" height="245" alt="Unknown" src="https://github.com/user-attachments/assets/9d43c500-d634-487f-93bf-e8450a1830ae" />

---

### Classifier-Free Guidance (CFG)

CFG (Ho & Salimans, 2022) steers generation toward a condition $c$ by 
extrapolating between conditional and unconditional noise predictions:

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + w\Big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)\Big)$$

where $w$ is the guidance scale — higher $w$ means stronger conditioning but less diversity.

### Joint Training Algorithm — Conditional + Unconditional.(used noise prediction  inplace for score pred)
<img width="666" height="570" alt="Screenshot 2026-03-20 at 3 13 32 PM" src="https://github.com/user-attachments/assets/bd10421a-dd55-46d0-8c3e-9a26c349de99" />


### CFG — Pros & Cons

| | |
|---|---|
| ✅ No separate classifier needed | ❌ Doubles inference cost (two forward passes) |
| ✅ Simple to implement | ❌ High $w$ causes oversaturation |
| ✅ Strong class-conditional control | ❌ Reduces sample diversity at high guidance |
| ✅ Works with any diffusion backbone | ❌ Requires condition dropout during training |

---

## Results

### Samples After CFG
samples for Hand,chestCT,breastMRI and CXR (Chest X-Ray)
<img width="543" height="192" alt="Unknown-2" src="https://github.com/user-attachments/assets/b6ed2a8e-7137-456d-a0fa-a2664fc837db" />

