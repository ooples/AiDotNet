---
title: "Causal Discovery"
description: "All 87 public types in the AiDotNet.causaldiscovery namespace, organized by kind."
section: "API Reference"
---

**87** public types in this namespace, organized by kind.

## Models & Types (76)

| Type | Summary |
|:-----|:--------|
| [`ANMAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/anmalgorithm/) | ANM (Additive Noise Model) — pairwise causal discovery via independence of residuals. |
| [`AVICIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/avicialgorithm/) | AVICI — Amortized Variational Inference for Causal Discovery. |
| [`AmortizedCDAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/amortizedcdalgorithm/) | Amortized Causal Discovery — meta-learning approach to causal structure learning. |
| [`BCDNetsAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/bcdnetsalgorithm/) | BCD-Nets — Bayesian Causal Discovery Networks. |
| [`BOSSAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/bossalgorithm/) | BOSS (Best Order Score Search) — efficient permutation-based structure learning. |
| [`BayesDAGAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/bayesdagalgorithm/) | BayesDAG — Bayesian DAG learning with gradient-based posterior inference. |
| [`CAMAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/camalgorithm/) | CAM (Causal Additive Model) — order-based causal discovery with additive nonparametric regression. |
| [`CAMUVAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/camuvalgorithm/) | CAM-UV — Causal Additive Model with Unobserved Variables. |
| [`CASTLEAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/castlealgorithm/) | CASTLE — Causal Structure Learning via neural networks with shared masked architecture. |
| [`CCDrAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/ccdralgorithm/) | CCDr (Concave penalized Coordinate Descent with reparameterization) for DAG learning. |
| [`CCMAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/ccmalgorithm/) | CCM — Convergent Cross-Mapping for detecting causation in nonlinear dynamical systems. |
| [`CDNODAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/cdnodalgorithm/) | CD-NOD — Constraint-based Discovery from Non-stationary / heterogeneous Data. |
| [`CGNNAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/cgnnalgorithm/) | CGNN — Causal Generative Neural Networks. |
| [`CORLAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/corlalgorithm/) | CORL — Causal Ordering via Reinforcement Learning. |
| [`CPCAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/cpcalgorithm/) | CPC (Conservative PC) — PC variant that avoids erroneous v-structure orientation. |
| [`CausalDiscoveryResult<T>`](/docs/reference/wiki/causaldiscovery/causaldiscoveryresult/) | Contains the results of a causal discovery analysis, including the learned graph and convergence metrics. |
| [`CausalDiscoverySelector<T>`](/docs/reference/wiki/causaldiscovery/causaldiscoveryselector/) | Feature selector that uses any causal discovery algorithm to select features based on causal relationships. |
| [`CausalGraph<T>`](/docs/reference/wiki/causaldiscovery/causalgraph/) | Represents a causal Directed Acyclic Graph (DAG) discovered from observational data. |
| [`CausalVAEAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/causalvaealgorithm/) | CausalVAE — Causal Variational Autoencoder. |
| [`DAGGNNAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/daggnnalgorithm/) | DAG-GNN — DAG Structure Learning with Graph Neural Networks. |
| [`DAGMALinear<T>`](/docs/reference/wiki/causaldiscovery/dagmalinear/) | DAGMA Linear — DAG learning via M-matrices and a log-determinant acyclicity characterization. |
| [`DAGMANonlinear<T>`](/docs/reference/wiki/causaldiscovery/dagmanonlinear/) | DAGMA Nonlinear — DAG learning via M-matrices and log-determinant with MLP structural equations. |
| [`DECIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/decialgorithm/) | DECI — Deep End-to-end Causal Inference. |
| [`DYNOTEARSAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/dynotearsalgorithm/) | DYNOTEARS — Dynamic NOTEARS for time series structure learning. |
| [`DiBSAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/dibsalgorithm/) | DiBS — Differentiable Bayesian Structure Learning. |
| [`DirectLiNGAMAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/directlingamalgorithm/) | DirectLiNGAM — direct method for LiNGAM without ICA. |
| [`ExactSearchAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/exactsearchalgorithm/) | Exact Search (Dynamic Programming) — optimal DAG structure learning. |
| [`FCIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/fcialgorithm/) | FCI (Fast Causal Inference) — constraint-based discovery with latent confounders. |
| [`FGESAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/fgesalgorithm/) | FGES (Fast Greedy Equivalence Search) — greedy DAG search with BIC score caching. |
| [`FastIAMBAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/fastiambalgorithm/) | Fast-IAMB — faster variant of IAMB using speculative forward selection. |
| [`GAEAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/gaealgorithm/) | GAE — Graph Autoencoder for causal discovery. |
| [`GESAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/gesalgorithm/) | GES (Greedy Equivalence Search) — score-based causal discovery over equivalence classes. |
| [`GFCIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/gfcialgorithm/) | GFCI — Greedy FCI, a hybrid of GES and FCI. |
| [`GOBNILPAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/gobnilpalgorithm/) | GOBNILP — Globally Optimal Bayesian Network learning using Integer Linear Programming. |
| [`GOLEMAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/golemalgorithm/) | GOLEM — Gradient-based Optimization with Likelihood for structure learning of linear DAGs. |
| [`GRaSPAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/graspalgorithm/) | GRaSP (Greedy Relaxation of Sparsest Permutation) — permutation-based causal discovery. |
| [`GraNDAGAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/grandagalgorithm/) | GraN-DAG — Gradient-based Neural DAG Learning. |
| [`GrangerCausalityAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/grangercausalityalgorithm/) | Granger Causality — time series causal discovery via predictive improvement. |
| [`H2PCAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/h2pcalgorithm/) | H2PC — Hybrid HPC (Hybrid Parents and Children) algorithm. |
| [`HillClimbingAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/hillclimbingalgorithm/) | Hill Climbing — greedy score-based DAG structure learning. |
| [`IAMBAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/iambalgorithm/) | IAMB (Incremental Association Markov Blanket) — efficient Markov blanket discovery. |
| [`ICALiNGAMAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/icalingamalgorithm/) | ICA-LiNGAM — Linear Non-Gaussian Acyclic Model using Independent Component Analysis. |
| [`IGCIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/igcialgorithm/) | IGCI (Information-Geometric Causal Inference) — bivariate causal discovery via entropy. |
| [`InterventionalDistribution<T>`](/docs/reference/wiki/causaldiscovery/interventionaldistribution/) | Represents the interventional distribution P(Y \| do(X = x)) from Pearl's do-calculus. |
| [`IterativeMCMCAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/iterativemcmcalgorithm/) | Iterative MCMC — iteratively refined MCMC for Bayesian network structure learning. |
| [`K2Algorithm<T>`](/docs/reference/wiki/causaldiscovery/k2algorithm/) | K2 Algorithm — score-based learning with known variable ordering. |
| [`KraskovMIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/kraskovmialgorithm/) | Kraskov MI — Mutual Information estimation using k-nearest neighbors (KSG estimator). |
| [`LPCMCIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/lpcmcialgorithm/) | LPCMCI — Latent PCMCI for time series with hidden confounders. |
| [`MCSLAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/mcslalgorithm/) | MCSL — Masked Gradient-Based Causal Structure Learning. |
| [`MMHCAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/mmhcalgorithm/) | MMHC — Max-Min Hill-Climbing, a hybrid constraint-based + score-based algorithm. |
| [`MMPCAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/mmpcalgorithm/) | MMPC (Max-Min Parents and Children) — identifies the parents and children of each variable. |
| [`MarkovBlanketAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/markovblanketalgorithm/) | Markov Blanket (Grow-Shrink) Algorithm — discovers the Markov blanket of each variable. |
| [`NOTEARSLinear<T>`](/docs/reference/wiki/causaldiscovery/notearslinear/) | NOTEARS Linear — continuous optimization for DAG structure learning with linear relationships. |
| [`NOTEARSLowRank<T>`](/docs/reference/wiki/causaldiscovery/notearslowrank/) | NOTEARS Low-Rank — DAG learning with low-rank parameterization for scalability. |
| [`NOTEARSNonlinear<T>`](/docs/reference/wiki/causaldiscovery/notearsnonlinear/) | NOTEARS Nonlinear — continuous optimization for DAG structure learning with nonlinear (MLP) relationships. |
| [`NOTEARSSobolev<T>`](/docs/reference/wiki/causaldiscovery/notearssobolev/) | NOTEARS with Sobolev regularization — DAG learning with smoothness constraints. |
| [`NTSNOTEARSAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/ntsnotearsalgorithm/) | NTS-NOTEARS — Nonstationary Time Series NOTEARS. |
| [`NeuralGrangerAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/neuralgrangeralgorithm/) | Neural Granger Causality — deep learning extension of Granger causality. |
| [`NoCurlAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/nocurlalgorithm/) | NoCurl — DAG learning via curl-free constraints on the graph structure. |
| [`OCSEAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/ocsealgorithm/) | oCSE — optimal Causation Entropy for causal network inference. |
| [`OrderMCMCAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/ordermcmcalgorithm/) | Order MCMC — MCMC sampling over variable orderings for Bayesian structure learning. |
| [`PCAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/pcalgorithm/) | PC Algorithm — constraint-based causal discovery using conditional independence tests. |
| [`PCMCIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/pcmcialgorithm/) | PCMCI — PC algorithm for Momentary Conditional Independence in time series. |
| [`PCMCIPlusAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/pcmciplusalgorithm/) | PCMCI+ — extension of PCMCI that also discovers contemporaneous causal links. |
| [`PCNOTEARSAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/pcnotearsalgorithm/) | PC-NOTEARS — Hybrid of PC skeleton discovery with NOTEARS continuous optimization. |
| [`PNLAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/pnlalgorithm/) | PNL (Post-Nonlinear Causal Model) — Y = g(f(X) + N). |
| [`PartitionMCMCAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/partitionmcmcalgorithm/) | Partition MCMC — MCMC sampling over DAG partitions for structure learning. |
| [`RCDAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/rcdalgorithm/) | RCD (Repetitive Causal Discovery) — LiNGAM extension for latent confounders. |
| [`RFCIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/rfcialgorithm/) | RFCI (Really Fast Causal Inference) — scalable FCI for large datasets. |
| [`RSMAX2Algorithm<T>`](/docs/reference/wiki/causaldiscovery/rsmax2algorithm/) | RSMAX2 — Restricted Maximization, a hybrid constraint-based + score-based algorithm. |
| [`TCDFAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/tcdfalgorithm/) | TCDF — Temporal Causal Discovery Framework. |
| [`TSFCIAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/tsfcialgorithm/) | tsFCI — time series Fast Causal Inference. |
| [`TabuSearchAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/tabusearchalgorithm/) | Tabu Search — score-based DAG learning with memory to escape local optima. |
| [`TiMINoAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/timinoalgorithm/) | TiMINo — Time series Models with Independent Noise. |
| [`TransferEntropyAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/transferentropyalgorithm/) | Transfer Entropy — information-theoretic measure of directed information flow. |
| [`VARLiNGAMAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/varlingamalgorithm/) | VAR-LiNGAM — Vector Autoregressive LiNGAM for time series causal discovery. |

## Base Classes (10)

| Type | Summary |
|:-----|:--------|
| [`BayesianCausalBase<T>`](/docs/reference/wiki/causaldiscovery/bayesiancausalbase/) | Base class for Bayesian causal discovery algorithms. |
| [`CausalDiscoveryBase<T>`](/docs/reference/wiki/causaldiscovery/causaldiscoverybase/) | Abstract base class for causal discovery algorithms with shared statistical utilities. |
| [`ConstraintBasedBase<T>`](/docs/reference/wiki/causaldiscovery/constraintbasedbase/) | Base class for constraint-based causal discovery algorithms (PC, FCI, MMPC, etc.). |
| [`ContinuousOptimizationBase<T>`](/docs/reference/wiki/causaldiscovery/continuousoptimizationbase/) | Base class for continuous optimization causal discovery methods (NOTEARS, DAGMA, GOLEM). |
| [`DeepCausalBase<T>`](/docs/reference/wiki/causaldiscovery/deepcausalbase/) | Base class for deep learning-based causal discovery algorithms. |
| [`FunctionalBase<T>`](/docs/reference/wiki/causaldiscovery/functionalbase/) | Base class for functional/ICA-based causal discovery algorithms (LiNGAM, ANM, etc.). |
| [`HybridBase<T>`](/docs/reference/wiki/causaldiscovery/hybridbase/) | Base class for hybrid causal discovery algorithms that combine constraint-based and score-based methods. |
| [`InfoTheoreticBase<T>`](/docs/reference/wiki/causaldiscovery/infotheoreticbase/) | Base class for information-theoretic causal discovery algorithms. |
| [`ScoreBasedBase<T>`](/docs/reference/wiki/causaldiscovery/scorebasedbase/) | Base class for score-based causal discovery algorithms (GES, FGES, Hill Climbing, Tabu, etc.). |
| [`TimeSeriesCausalBase<T>`](/docs/reference/wiki/causaldiscovery/timeseriescausalbase/) | Base class for time series causal discovery algorithms (Granger, PCMCI, DYNOTEARS, etc.). |

## Interfaces (1)

| Type | Summary |
|:-----|:--------|
| [`ICausalDiscoveryAlgorithm<T>`](/docs/reference/wiki/causaldiscovery/icausaldiscoveryalgorithm/) | Interface for causal structure learning algorithms that discover Directed Acyclic Graphs (DAGs) from data. |

