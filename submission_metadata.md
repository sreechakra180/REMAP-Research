# Submission Metadata

**Title:**  
A Recursively Adaptive Meta-Plasticity Framework with Lyapunov-Constrained Stability

**Authors:**  
Sree Chakra Reddy (Dual Degree Student from IIT M, IIIT RKV)
*[Add any co-authors here]*

**Corresponding Author Email:**  
*[Insert Email]*

**Abstract:**  
Existing meta-learning systems use fixed update rules and lack bounded recursive adaptation, resulting in severe catastrophic forgetting and unbounded divergence during higher-order gradient computation on sequential tasks. We resolve these instabilities through a hierarchically decoupled architecture encompassing three learning layers (Object Level, Meta-Plasticity, and Epistemic Recursion), governed by a strict Lyapunov stability bound enforced by a novel Stability Guardian module. Furthermore, an Autonomous Abstraction Formation module consolidates memory sequences using a Task Coherence Regularizer. Extensive experiments on Omniglot and Split-CIFAR100 demonstrate that our framework achieves stable, bounded adaptation without divergence, significantly outperforms state-of-the-art baselines in few-shot adaptation accuracy, and sharply reduces catastrophic forgetting during continual learning.

**Keywords:**  
Meta-Learning, Continual Learning, Catastrophic Forgetting, Lyapunov Stability, Higher-Order Optimization, Adaptive Plasticity

**Code Repository:**  
*[Insert Link to GitHub Repository if applicable, or indicate that code is included in supplementary materials]*
