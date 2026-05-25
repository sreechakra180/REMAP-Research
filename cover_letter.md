Dear Editor/Area Chair,

Please find enclosed our manuscript entitled **"A Recursively Adaptive Meta-Plasticity Framework with Lyapunov-Constrained Stability"** for consideration for publication.

In this work, we address a fundamental issue in meta-learning systems: severe catastrophic forgetting and unbounded divergence during higher-order gradient computations on sequential tasks. Existing meta-learning architectures typically employ fixed update rules and lack bounded recursive adaptation, severely limiting their applicability in continual and unbounded learning environments.

To resolve these instabilities, we introduce a hierarchically decoupled architecture encompassing three learning layers (F0, F1, F2). Crucially, the updates within this recursive system are governed by a strict Lyapunov stability bound enforced by a novel Stability Guardian module. Furthermore, we consolidate memory sequences via an Autonomous Abstraction Formation (AAF) module with a Task Coherence Regularizer (TCR).

Our empirical results demonstrate that our framework achieves stable, bounded adaptation without gradient divergence. On the Omniglot adaptation benchmark, our framework exceeds state-of-the-art accuracy, and on the Split-CIFAR100 continual learning benchmark, it retains greater than 75% accuracy over 10 sequential tasks—significantly reducing forgetting compared to MAML and standard SGD. We also supply a comprehensive ablation study and a fully reproducible codebase. 

We confirm that this manuscript has not been published elsewhere and is not under consideration by another journal or conference. All authors have approved the manuscript and agree with its submission.

Thank you for your time and consideration of our work.

Sincerely,

**Sree Chakra Reddy**
Dual Degree Student
IIT M, IIIT RKV

