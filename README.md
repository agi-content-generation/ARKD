# ARKD: Adaptive Reinforcement Learning-Guided Bidirectional KL Divergence Distillation for Text Generation

## 📖 Overview

As large-scale pretrained language models (LLMs) proliferate, deploying them efficiently requires compression techniques that preserve both generation quality and generalization.

**ARKD** is a reinforcement learning–driven knowledge distillation framework that:

- Theoretically analyzes the complementarity of **forward KL (FKL)** and **reverse KL (RKL)** in aligning teacher–student distributions.
- Introduces a **policy network** to dynamically weight FKL and RKL via reinforcement learning.
- Achieves superior performance compared to static KL-based and existing KD baselines.

Experiments show **1%–4% improvements** in text generation quality and generalization across multiple LLMs and benchmarks.





------

## 🚧 Code Availability

We are currently **cleaning and organizing the code**.
 The implementation will be uploaded to this repository soon.

In the meantime, you may refer to the paper for details of:

- Theoretical analysis of FKL/RKL
- Reinforcement learning–guided adaptive KL weighting
- Experimental setup and results

📅 **Expected release date of the code:** *[to be confirmed]*

------

## ⚡ Usage (Coming Soon)

When the code is released, we will provide:

- Installation instructions
- Example training scripts for distillation
- Configuration files for reproducing experiments
- Evaluation scripts on benchmarks (Dolly, SelfInst, SuperNatural-Instructions, UnNI, VicunaEval)

Stay tuned!

------

## 📊 Main Results

- On both **in-distribution** (DollyEval) and **out-of-distribution** datasets (SelfInst, SuperNatural-Instructions, UnNI, VicunaEval), ARKD outperforms:
  - Standard supervised fine-tuning (SFT)
  - Sequence-level KD
  - FKL / RKL individually
  - Static FKL+RKL combinations

------

## 📬 Contact

For questions or collaborations, please reach out to the authors via email (see paper).
