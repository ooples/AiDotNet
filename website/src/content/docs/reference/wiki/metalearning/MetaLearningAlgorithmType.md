---
title: "MetaLearningAlgorithmType"
description: "Specifies the type of meta-learning algorithm used for few-shot learning and quick adaptation."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning`

Specifies the type of meta-learning algorithm used for few-shot learning and quick adaptation.

## For Beginners

Meta-learning algorithms are designed to "learn how to learn."
Instead of learning a single task, they learn to quickly adapt to new tasks with minimal data.
This enum lists all supported meta-learning algorithms in the framework.

## How It Works

**Algorithm Categories:**

- **Optimization-based:** MAML, Reptile, Meta-SGD, iMAML, ANIL, BOIL, LEO
- **Metric-based:** ProtoNets, MatchingNetworks, RelationNetwork, TADAM
- **Memory-based:** MANN, NTM
- **Hybrid/Advanced:** CNAP, SEAL, GNNMeta, MetaOptNet
- **Neural Processes:** CNP, NP, ANP, ConvCNP, ConvNP, TNP, SwinTNP, TETNP, EquivCNP, SteerCNP, RCNP, LBANP
- **Foundation Model Era:** MetaLoRA, LoRARecycle, ICMFusion, MetaLoRABank, AutoLoRA, MetaDiff, MetaDM, MetaDDPM
- **Bayesian Extensions:** PACOH, MetaPACOH, BMAML, BayProNet, FlexPACBayes
- **Cross-Domain:** MetaFDMixup, FreqPrior, MetaCollaborative, SDCL, FreqPrompt, OpenMAMLPlus
- **Meta-RL:** PEARL, DREAM, DiscoRL, InContextRL, HyperNetMetaRL, ContextMetaRL
- **Continual/Online:** ACL, iTAML, MetaContinualAL, MePo, OML, MOCA
- **Task Augmentation:** MetaTask, ATAML, MPTS, DynamicTaskSampling, UnsupervisedMetaLearn
- **Transductive:** GCDPLNet, BayTransProto, JMP, ETPN, ActiveTransFSL
- **Hypernetwork:** TaskCondHyperNet, HyperCLIP, RecurrentHyperNet, HyperNeRFMeta

## Fields

| Field | Summary |
|:-----|:--------|
| `ACL` | ACL - Adaptive Continual Learning with meta-learning (2024). |
| `ANIL` | Almost No Inner Loop (Raghu et al., 2020). |
| `ANP` | Attentive Neural Process (Kim et al., ICLR 2019). |
| `ATAML` | ATAML - Adaptive Task Augmentation for Meta-Learning (2024). |
| `ActiveTransFSL` | ActiveTransFSL - Active Transductive Few-Shot Learning (2024). |
| `AutoLoRA` | AutoLoRA - Automatic LoRA rank and configuration selection (2024). |
| `BMAML` | BMAML - Bayesian MAML (Yoon et al., NeurIPS 2018). |
| `BOIL` | Body Only Inner Loop (Oh et al., 2021). |
| `BayProNet` | BayProNet - Bayesian Prototypical Networks (2024). |
| `BayTransProto` | BayTransProto - Bayesian Transductive Prototypical Networks (2024). |
| `CAML` | CAML - Context-Aware Meta-Learning (Fifty et al., NeurIPS 2023). |
| `CAVIA` | Fast Context Adaptation via Meta-Learning (Zintgraf et al., ICML 2019). |
| `CNAP` | Conditional Neural Adaptive Processes (Requeima et al., 2019). |
| `CNP` | Conditional Neural Process (Garnelo et al., ICML 2018). |
| `ConstellationNet` | ConstellationNet - Structured part-based few-shot learning (Xu et al., ICLR 2021). |
| `ContextMetaRL` | Context Meta-RL - Context-based Meta-Reinforcement Learning (2024). |
| `ConvCNP` | Convolutional Conditional Neural Process (Gordon et al., ICLR 2020). |
| `ConvNP` | Convolutional Neural Process (Foong et al., 2020). |
| `DKT` | DKT - Deep Kernel Transfer (Patacchiola et al., ICLR 2020). |
| `DPGN` | DPGN - Distribution Propagation Graph Network (Yang et al., CVPR 2020). |
| `DREAM` | DREAM - Decoupled Reward-Environment Adaptation Meta-learning (2022). |
| `DeepEMD` | DeepEMD - Earth Mover's Distance for few-shot learning (Zhang et al., CVPR 2020). |
| `DiscoRL` | DiscoRL - Discovering Meta-RL Objectives (2024). |
| `DynamicTaskSampling` | Dynamic Task Sampling - Adaptive task distribution during meta-training (2024). |
| `EPNet` | EPNet - Embedding Propagation Network (Rodriguez et al., CVPR 2020). |
| `ETPN` | ETPN - Enhanced Transductive Prototypical Networks (2024). |
| `EquivCNP` | Equivariant Conditional Neural Process (Kawano et al., 2021). |
| `FEAT` | FEAT - Few-shot Embedding Adaptation with Transformer (Ye et al., CVPR 2020). |
| `FRN` | FRN - Few-shot Classification via Feature Map Reconstruction (Wertheimer et al., CVPR 2021). |
| `FewTURE` | FewTURE - Few-shot Transformer with Uncertainty and Reliable Estimation (Hiller et al., ECCV 2022). |
| `FlexPACBayes` | Flex-PAC-Bayes - Flexible PAC-Bayes bounds for meta-learning (2024). |
| `FreqPrior` | FreqPrior - Frequency-based Prior for cross-domain few-shot learning (2024). |
| `FreqPrompt` | FreqPrompt - Frequency-aware Prompt tuning for cross-domain FSL (2024). |
| `GCDPLNet` | GCDPLNet - Graph-based Class Distribution Propagation and Label Network (2024). |
| `GNNMeta` | Graph Neural Network for Meta-Learning. |
| `HyperCLIP` | HyperCLIP - Hypernetwork with CLIP-based task encoding (2024). |
| `HyperMAML` | HyperMAML - Hypernetwork-based MAML initialization. |
| `HyperNeRFMeta` | HyperNeRF Meta - Hypernetwork for Meta-learning Neural Radiance Fields (2024). |
| `HyperNetMetaRL` | HyperNet Meta-RL - Hypernetwork-based Meta-Reinforcement Learning (2024). |
| `HyperShot` | HyperShot - Kernel hypernetwork for few-shot learning. |
| `ICMFusion` | ICM Fusion - In-Context Model Fusion (2024). |
| `InContextRL` | In-Context RL - Reinforcement learning through in-context learning (2023). |
| `JMP` | JMP - Joint Meta-learning and Propagation for transductive FSL (2024). |
| `LBANP` | Latent Bottleneck Attentive Neural Process (Feng et al., ICML 2023). |
| `LEO` | Latent Embedding Optimization (Rusu et al., 2019). |
| `LaplacianShot` | LaplacianShot - Laplacian Regularized Few-Shot Learning (Ziko et al., ICML 2020). |
| `LoRARecycle` | LoRA-Recycle - Recycling LoRA adapters across tasks (2024). |
| `MAML` | Model-Agnostic Meta-Learning (Finn et al., 2017). |
| `MAMLPlusPlus` | MAML++ - How to Train Your MAML (Antoniou et al., ICLR 2019). |
| `MANN` | Memory-Augmented Neural Network (Santoro et al., 2016). |
| `MCL` | MCL - Meta-learning with Contrastive Learning. |
| `MOCA` | MOCA - Meta-learning Online Continual Adaptation (2024). |
| `MPTS` | MPTS - Meta-learning with Prioritized Task Sampling (2024). |
| `MatchingNetworks` | Matching Networks for One Shot Learning (Vinyals et al., 2016). |
| `MePo` | MePo - Meta-learning for Policy optimization in continual RL (2024). |
| `MetaBaseline` | Meta-Baseline - Simple pre-train then meta-train with cosine classifier (Chen et al., ICLR 2021). |
| `MetaCollaborative` | MetaCollaborative - Collaborative meta-learning across multiple source domains (2024). |
| `MetaContinualAL` | MetaContinualAL - Meta-learning for Continual Active Learning (2024). |
| `MetaDDPM` | MetaDDPM - Meta Denoising Diffusion Probabilistic Model (2024). |
| `MetaDM` | MetaDM - Meta Diffusion Model for few-shot generation (2024). |
| `MetaDiff` | MetaDiff - Meta-learning with Diffusion Models (2024). |
| `MetaFDMixup` | Meta-FDMixup - Feature Distribution Mixup for cross-domain few-shot learning (2021). |
| `MetaLoRA` | Meta-LoRA - Meta-learning with Low-Rank Adaptation (2024). |
| `MetaLoRABank` | Meta-LoRA Bank - Library of meta-learned LoRA modules (2024). |
| `MetaOptNet` | Meta-learning with differentiable convex optimization (Lee et al., 2019). |
| `MetaPACOH` | Meta-PACOH - Extended PACOH with hierarchical Bayesian meta-learning (2023). |
| `MetaSGD` | Meta-SGD with per-parameter learning rates (Li et al., 2017). |
| `MetaTask` | MetaTask - Meta-learning task generation for improved generalization (2024). |
| `NP` | Neural Process (Garnelo et al., 2018). |
| `NPBML` | NPBML - Neural Process-Based Meta-Learning. |
| `NTM` | Neural Turing Machine for meta-learning. |
| `OML` | OML - Online Meta-Learning (Javed & White, 2019). |
| `OpenMAML` | Open-MAML - MAML extended for open-set recognition. |
| `OpenMAMLPlus` | Open-MAML++ - Enhanced MAML for open-set cross-domain recognition (2024). |
| `PACOH` | PACOH - PAC-Bayesian Meta-Learning with Optimal Hyperparameters (Rothfuss et al., ICLR 2021). |
| `PEARL` | PEARL - Probabilistic Embeddings for Actor-critic RL (Rakelly et al., ICML 2019). |
| `PMF` | PMF - Pre-train, Meta-train, Fine-tune (Hu et al., ICLR 2022). |
| `PTMAP` | PT+MAP - Power Transform + Maximum A Posteriori (Hu et al., ICLR 2021). |
| `ProtoNets` | Prototypical Networks (Snell et al., 2017). |
| `R2D2` | R2-D2 - Meta-learning with Differentiable Closed-form Solvers (Bertinetto et al., ICLR 2019). |
| `RCNP` | Recurrent Conditional Neural Process (2024). |
| `RecurrentHyperNet` | Recurrent HyperNet - Hypernetwork with recurrent task encoding (2024). |
| `RelationNetwork` | Relation Network for few-shot learning (Sung et al., 2018). |
| `Reptile` | Reptile meta-learning algorithm (Nichol et al., 2018). |
| `SDCL` | SDCL - Self-Distillation Contrastive Learning for cross-domain FSL (2024). |
| `SEAL` | Self-Explanatory Attention Learning. |
| `SIB` | SIB - Sequential Information Bottleneck (Hu et al., 2020). |
| `SNAIL` | SNAIL - Simple Neural Attentive Meta-Learner (Mishra et al., ICLR 2018). |
| `SetFeat` | SetFeat - Matching Feature Sets for Few-Shot Classification (Afrasiyabi et al., CVPR 2022). |
| `SimpleShot` | SimpleShot - Nearest-centroid classification with feature normalization (Wang et al., 2019). |
| `SteerCNP` | Steerable Conditional Neural Process (Holderrieth et al., 2021). |
| `SwinTNP` | Swin Transformer Neural Process (2024). |
| `TADAM` | Task-Dependent Adaptive Metric (Oreshkin et al., 2018). |
| `TETNP` | Translation-Equivariant Transformer Neural Process (2024). |
| `TIM` | TIM - Transductive Information Maximization (Boudiaf et al., NeurIPS 2020). |
| `TNP` | Transformer Neural Process (Nguyen & Grover, ICML 2023). |
| `TaskCondHyperNet` | Task-Conditioned HyperNet - Hypernetwork conditioned on task representations (2024). |
| `UnsupervisedMetaLearn` | Unsupervised Meta-Learning - Meta-learning without task labels (Hsu et al., 2019). |
| `VERSA` | VERSA - Versatile and Efficient Few-shot Learning (Gordon et al., ICLR 2019). |
| `WarpGrad` | Warped Gradient Descent meta-learning (Flennerhag et al., ICLR 2020). |
| `iMAML` | Implicit MAML with implicit gradients (Rajeswaran et al., 2019). |
| `iTAML` | iTAML - Incremental Task-Agnostic Meta-Learning (Rajasegaran et al., 2020). |

