---
title: "CausalDiscoveryAlgorithmType"
description: "Specifies the algorithm to use for causal structure learning (DAG discovery)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the algorithm to use for causal structure learning (DAG discovery).

## For Beginners

These algorithms figure out which variables cause which other variables
by analyzing patterns in your data. Think of it like a detective figuring out cause-and-effect
relationships. Different algorithms are like different detective methods — some test independence
relationships, some optimize a score, and some use advanced math to find the best graph.

## How It Works

Causal discovery algorithms learn the causal structure (a Directed Acyclic Graph or DAG)
from observational data. Different algorithms make different assumptions about the data
(linearity, Gaussianity, faithfulness) and use different strategies (constraint testing,
score optimization, continuous optimization).

## Fields

| Field | Summary |
|:-----|:--------|
| `ANM` | ANM — Additive Noise Model. |
| `AVICI` | AVICI — Amortized Variational Inference for Causal Discovery. |
| `AmortizedCD` | Amortized Causal Discovery — meta-learning approach to causal discovery. |
| `BCDNets` | BCD-Nets — Scalable variational Bayesian Causal Discovery. |
| `BOSS` | BOSS — Bayesian Optimal Structure Search. |
| `BayesDAG` | BayesDAG — Bayesian DAG learning with direct parameterization. |
| `CAM` | CAM — Causal Additive Models. |
| `CAMUV` | CAM-UV — Causal Additive Models with Unobserved Variables. |
| `CASTLE` | CASTLE — Causal Structure Learning. |
| `CCDr` | CCDr — Concave penalized Coordinate Descent with reparameterization. |
| `CCM` | CCM — Convergent Cross-Mapping for detecting causality in dynamical systems. |
| `CDNOD` | CD-NOD — Causal Discovery from Nonstationary/heterogeneous Data. |
| `CGNN` | CGNN — Causal Generative Neural Networks. |
| `CORL` | CORL — Causal Order learning via Reinforcement Learning. |
| `CPC` | CPC — Conservative PC. |
| `CausalVAE` | CausalVAE — Variational Autoencoder for causal representation learning. |
| `DAGGNN` | DAG-GNN — Graph Neural Network for DAG structure learning. |
| `DAGMALinear` | DAGMA Linear — log-determinant acyclicity constraint via M-matrices. |
| `DAGMANonlinear` | DAGMA Nonlinear — extends DAGMA with neural network function approximation. |
| `DECI` | DECI — Deep End-to-end Causal Inference. |
| `DYNOTEARS` | DYNOTEARS — Dynamic NOTEARS for time series structure learning. |
| `DiBS` | DiBS — Differentiable Bayesian Structure Learning. |
| `DirectLiNGAM` | DirectLiNGAM — Direct method for LiNGAM without ICA. |
| `ExactSearch` | Exact Search — dynamic programming for exact structure learning (exponential complexity). |
| `FCI` | FCI — Fast Causal Inference. |
| `FGES` | FGES — Fast Greedy Equivalence Search. |
| `FastIAMB` | Fast-IAMB — Faster variant of IAMB. |
| `GAE` | GAE — Graph Autoencoder for structure learning. |
| `GES` | GES — Greedy Equivalence Search. |
| `GFCI` | GFCI — Greedy FCI. |
| `GOBNILP` | GOBNILP — Integer Linear Programming for exact Bayesian network structure learning. |
| `GOLEM` | GOLEM — likelihood-based single-loop optimization without augmented Lagrangian. |
| `GRaSP` | GRaSP — Greedy Relaxation of the Sparsest Permutation. |
| `GraNDAG` | GraNDAG — Gradient-based Neural DAG Learning. |
| `GrangerCausality` | Granger Causality — tests whether one time series helps predict another. |
| `H2PC` | H2PC — Hybrid HPC algorithm. |
| `HillClimbing` | Hill Climbing — greedy local search with BIC or BDeu scoring. |
| `IAMB` | IAMB — Incremental Association Markov Blanket. |
| `ICALiNGAM` | ICA-LiNGAM — Linear Non-Gaussian Acyclic Model using Independent Component Analysis. |
| `IGCI` | IGCI — Information-Geometric Causal Inference. |
| `IterativeMCMC` | Iterative MCMC — Iterative Bayesian structure learning. |
| `K2` | K2 Algorithm — score-based search with a known variable ordering. |
| `KraskovMI` | Kraskov Mutual Information — k-nearest neighbor mutual information estimator. |
| `LPCMCI` | LPCMCI — Latent PCMCI for time series with latent confounders. |
| `MCSL` | MCSL — Multi-scale Causal Structure Learning. |
| `MMHC` | MMHC — Max-Min Hill Climbing. |
| `MMPC` | MMPC — Max-Min Parents and Children. |
| `MarkovBlanket` | Markov Blanket discovery via the Grow-Shrink algorithm. |
| `NOTEARSLinear` | NOTEARS Linear — continuous optimization with tr(e^(W∘W))-d acyclicity constraint. |
| `NOTEARSLowRank` | NOTEARS Low-Rank — low-rank approximation for scalability to high dimensions. |
| `NOTEARSNonlinear` | NOTEARS Nonlinear — extends NOTEARS with MLP (multi-layer perceptron) for nonlinear relationships. |
| `NOTEARSSobolev` | NOTEARS with Sobolev basis functions for nonlinear relationships. |
| `NTSNOTEARS` | NTS-NOTEARS — Non-stationary Time Series NOTEARS. |
| `NeuralGranger` | Neural Granger Causality — deep learning extension of Granger causality. |
| `NoCurl` | NoCurl — curl-free constraint for acyclicity. |
| `OCSE` | oCSE — Optimal Causation Entropy for detecting causal relationships. |
| `OrderMCMC` | Order MCMC — MCMC over topological orderings for Bayesian structure learning. |
| `PC` | PC Algorithm — the gold standard constraint-based method using conditional independence tests. |
| `PCMCI` | PCMCI — PC algorithm adapted for time series with momentary conditional independence. |
| `PCMCIPlus` | PCMCI+ — Extension of PCMCI that also discovers contemporaneous causal links. |
| `PCNOTEARS` | PC-NOTEARS — Hybrid combining PC skeleton with NOTEARS optimization. |
| `PNL` | PNL — Post-Nonlinear causal model. |
| `PartitionMCMC` | Partition MCMC — MCMC over DAG partitions. |
| `RCD` | RCD — Repetitive Causal Discovery. |
| `RFCI` | RFCI — Really Fast Causal Inference. |
| `RSMAX2` | RSMAX2 — Restricted maximization algorithm. |
| `TCDF` | TCDF — Temporal Causal Discovery Framework. |
| `TSFCI` | tsFCI — Time series Fast Causal Inference. |
| `TabuSearch` | Tabu Search — hill climbing with a tabu list to escape local optima. |
| `TiMINo` | TiMINo — Time series Model with Independent Noise. |
| `TransferEntropy` | Transfer Entropy — information-theoretic measure of directed information flow. |
| `VARLiNGAM` | VAR-LiNGAM — LiNGAM for time series via Vector Autoregressive model. |

