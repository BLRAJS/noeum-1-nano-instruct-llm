<p align="center">
  <img src="https://noeum.ai/wp-content/uploads/2025/11/noeum.png" alt="Noeum.ai" width="140" />
</p>

# Noeum-1-Nano (Post-trained) — Chat, Think Mode & Reproducible Evals

This repository provides a **user-friendly toolkit** for running the **post-trained** Noeum-1-Nano model (chat + reasoning), including:
- Interactive chat with `/think` on/off
- Controlled **thinking budget** and decoding sweeps
- Logging and reproducible evaluation presets

**Links**
- Website: https://noeum.ai  
- Model card + weights: https://huggingface.co/noeum/noeum-1-nano  

---

## About Noeum.ai

**Noeum.ai** is an independent AI research & engineering lab based in **Vienna, Austria**, focused on **efficiency-first training** and **reproducible evaluation**.  
Founded and led by **Bledar Ramo**, Noeum.ai builds models end-to-end—from **pre-training** to **post-training** and **benchmarking**—with the goal of advancing **novel reasoning capabilities** while improving reasoning quality **per unit of compute**.

**Core philosophy:** iterate fast at nano scale, validate what truly works, then scale only proven techniques.

---

## What is Noeum-1-Nano?

**Noeum-1-Nano** is a nano-scale **Mixture-of-Experts (MoE)** language model:
- **Size:** 0.6B total parameters / ~0.2B active parameters  
- **Training:** 18B high-signal tokens  
- **From scratch:** no pretrained weights, no inherited checkpoints  
- **Reasoning control:** optional **System-2 style** `/think` mode with a configurable reasoning budget  

The public release emphasizes **fair and reproducible evaluation**: baseline benchmarks are reported with **think mode disabled**, to avoid inflating comparisons via extra reasoning tokens.

### Why it matters (in one paragraph)

Most small-model baselines are trained on far larger token budgets. Noeum-1-Nano is a proof-of-method that careful architecture + high-signal data can produce **competitive nano-class behavior** under tight constraints—then use the same workflow to scale validated recipes to larger models (multimodal + multilingual) when compute allows.

---
