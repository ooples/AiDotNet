# Exhaustive Feature Selection Methods Catalog

This document catalogs all known feature selection methods for implementation in AiDotNet.

## Current Implementation Status

| Category | Method | Status |
|----------|--------|--------|
| **Filter - Univariate** | Chi-Squared | ✅ Implemented |
| **Filter - Univariate** | ANOVA F-Value | ✅ Implemented |
| **Filter - Univariate** | Mutual Information | ✅ Implemented |
| **Filter - Univariate** | Variance Threshold | ✅ Implemented |
| **Filter - Multivariate** | Correlation-based | ✅ Implemented |
| **Wrapper** | Forward Selection | ✅ Implemented |
| **Wrapper** | Backward Elimination | ✅ Implemented |
| **Wrapper** | Recursive Feature Elimination (RFE) | ✅ Implemented |
| **Embedded** | SelectFromModel | ✅ Implemented |

---

## 1. FILTER METHODS - STATISTICAL TESTS

### 1.1 Parametric Tests
| Method | Description | Priority |
|--------|-------------|----------|
| Student's t-test | Two-sample mean comparison | High |
| Welch's t-test | Unequal variance t-test | High |
| Paired t-test | Paired samples comparison | Medium |
| One-way ANOVA | Multi-group mean comparison | ✅ Done |
| Two-way ANOVA | Two-factor analysis | Medium |
| MANOVA | Multivariate ANOVA | Low |
| ANCOVA | ANOVA with covariates | Low |
| Hotelling's T² | Multivariate t-test | Low |
| Z-test | Large sample mean test | Medium |
| F-test | Variance ratio test | Medium |
| Bartlett's test | Homogeneity of variances | Low |
| Levene's test | Robust variance homogeneity | Low |
| Brown-Forsythe test | Median-based Levene | Low |

### 1.2 Non-Parametric Tests
| Method | Description | Priority |
|--------|-------------|----------|
| Mann-Whitney U | Non-parametric two-sample | High |
| Wilcoxon signed-rank | Paired non-parametric | High |
| Kruskal-Wallis H | Non-parametric ANOVA | High |
| Friedman test | Non-parametric repeated measures | Medium |
| Mood's median test | Median comparison | Low |
| Kolmogorov-Smirnov | Distribution comparison | Medium |
| Anderson-Darling | Distribution test | Low |
| Shapiro-Wilk | Normality test for filtering | Low |
| Runs test | Randomness test | Low |
| Sign test | Direction of differences | Low |

### 1.3 Categorical Tests
| Method | Description | Priority |
|--------|-------------|----------|
| Chi-squared test | ✅ Already implemented | Done |
| G-test (likelihood ratio) | Alternative to chi-squared | Medium |
| Fisher's exact test | Small sample categorical | High |
| Barnard's test | Unconditional exact test | Low |
| McNemar's test | Paired categorical | Medium |
| Cochran's Q test | Multiple paired categorical | Low |
| Bowker's test | Symmetry test | Low |
| Stuart-Maxwell test | Marginal homogeneity | Low |

### 1.4 Correlation-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Pearson correlation | ✅ Already implemented | Done |
| Spearman rank correlation | Monotonic relationships | High |
| Kendall tau-a | Rank correlation | High |
| Kendall tau-b | Handles ties | High |
| Kendall tau-c | Rectangular tables | Low |
| Point-biserial correlation | Binary vs continuous | High |
| Biserial correlation | Dichotomized continuous | Medium |
| Polyserial correlation | Ordinal vs continuous | Low |
| Polychoric correlation | Two ordinal variables | Low |
| Tetrachoric correlation | Two binary variables | Low |
| Partial correlation | Controlling for variables | High |
| Semi-partial correlation | Part correlation | Medium |
| Distance correlation | Non-linear relationships | High |
| Brownian correlation | Brownian motion based | Low |
| RV coefficient | Multivariate correlation | Low |
| Canonical correlation | Between variable sets | Medium |
| Copula-based correlation | Dependency structure | Low |
| Hoeffding's D | Independence measure | Medium |
| Schweizer-Wolff | Copula-based dependence | Low |

---

## 2. FILTER METHODS - INFORMATION THEORY

### 2.1 Entropy-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Mutual Information | ✅ Already implemented | Done |
| Normalized MI (NMI) | Scaled mutual information | High |
| Adjusted MI (AMI) | Chance-corrected MI | High |
| Standardized MI (SMI) | Z-score normalized | Medium |
| Symmetric Uncertainty | Normalized MI variant | High |
| Information Gain | Entropy reduction | High |
| Gain Ratio | Normalized information gain | High |
| Intrinsic Value | Split information | Medium |
| Gini Index | Impurity measure | High |
| Entropy | Shannon entropy | High |
| Rényi entropy | Generalized entropy | Low |
| Tsallis entropy | Non-extensive entropy | Low |
| Kolmogorov complexity | Algorithmic information | Low |
| Conditional entropy | H(Y|X) | High |
| Joint entropy | H(X,Y) | High |
| Cross entropy | Between distributions | Medium |
| KL divergence | Distribution difference | High |
| JS divergence | Symmetric KL | Medium |
| Total correlation | Multivariate MI | Medium |
| Dual total correlation | Alternative multivariate | Low |
| Interaction information | Three-way MI | Medium |
| Co-information | Generalized interaction | Low |

### 2.2 Multivariate Information-Theoretic
| Method | Description | Priority |
|--------|-------------|----------|
| mRMR (Min Redundancy Max Relevance) | Classic multivariate MI | **Critical** |
| mRMR-D | Difference variant | High |
| mRMR-Q | Quotient variant | High |
| MIFS (MI Feature Selection) | Original MI-based | High |
| MIFS-U | Updated MIFS | Medium |
| JMI (Joint MI) | Joint information | High |
| JMIM (Joint MI Maximization) | JMI variant | Medium |
| CMIM (Conditional MI Max) | Conditional approach | High |
| CIFE (Conditional Infomax) | Feature extraction | Medium |
| ICAP (Interaction Capping) | Capped interactions | Medium |
| DISR (Double Input Sym Relevance) | Symmetric MIFS | Medium |
| IGFS (Info Gain Feature Selection) | IG-based multivariate | Medium |
| FCBF (Fast Correlation-Based) | Fast symmetrical uncertainty | High |
| CFS (Correlation-based FS) | Subset evaluation | High |
| INTERACT | Interaction-based | Medium |
| CMI (Conditional MI) | Full conditional | High |
| PMI (Pointwise MI) | Instance-level | Medium |
| NPMI (Normalized PMI) | Normalized pointwise | Medium |
| IWFS (Interaction Weight FS) | Weighted interactions | Low |
| LSMI (Least Squares MI) | LS estimation of MI | Medium |
| HSIC (Hilbert-Schmidt Independence) | Kernel-based independence | High |
| BAHSIC (Backward HSIC) | HSIC backward elimination | Medium |
| FOHSIC (Forward HSIC) | HSIC forward selection | Medium |

---

## 3. FILTER METHODS - DISTANCE/SIMILARITY BASED

### 3.1 Relief Family
| Method | Description | Priority |
|--------|-------------|----------|
| Relief | Original algorithm | High |
| ReliefF | Handles multi-class, missing values | **Critical** |
| RReliefF | Regression Relief | High |
| SURF (Spatially Uniform ReliefF) | Distance-weighted | High |
| SURF* | Enhanced SURF | Medium |
| MultiSURF | Multi-threshold SURF | High |
| MultiSURF* | Enhanced MultiSURF | Medium |
| TuRF (Tuned ReliefF) | Iterative Relief | Medium |
| VLSReliefF | Very Large Scale | Medium |
| ReliefSeq | For sequencing data | Low |
| SWRF (Sigmoid Weighted ReliefF) | Sigmoid weighting | Low |
| I-RELIEF | Iterative Relief | Medium |
| Online Relief | Streaming version | Medium |
| Boosted ReliefF | Ensemble Relief | Medium |
| EC-ReliefF | Evaporative Cooling ReliefF | Low |

### 3.2 Other Distance-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Fisher Score | Between/within class ratio | **Critical** |
| Fisher's Linear Discriminant | LDA-based | High |
| Separability Index | Class separation | Medium |
| Bhattacharyya Distance | Distribution overlap | Medium |
| Mahalanobis Distance | Covariance-weighted | Medium |
| Hellinger Distance | Square root of distribution diff | Medium |
| Wasserstein Distance | Earth mover's distance | Medium |
| Energy Distance | Statistical distance | Low |
| Maximum Mean Discrepancy (MMD) | Kernel-based distance | Medium |
| Chi-squared Distance | Histogram distance | Medium |
| Euclidean Distance Ratio | Simple distance ratio | Low |
| Cosine Similarity Score | Angular similarity | Medium |
| Jaccard Similarity | Set overlap | Medium |
| Dice Coefficient | Set similarity | Low |
| Tanimoto Coefficient | Extended Jaccard | Low |

---

## 4. FILTER METHODS - SPECTRAL/GRAPH BASED

### 4.1 Spectral Methods
| Method | Description | Priority |
|--------|-------------|----------|
| Laplacian Score | Locality preserving | High |
| SPEC (Spectral FS) | Eigenvector-based | High |
| Spectral Clustering FS | Cluster structure | Medium |
| Trace Ratio | Trace optimization | Medium |
| Graph Laplacian FS | Graph-based Laplacian | Medium |
| Normalized Cut FS | Graph cut | Low |
| Min-Cut FS | Minimum cut | Low |
| Ratio Cut FS | Ratio-based cut | Low |
| Multi-Cluster FS (MCFS) | Multiple clusters | Medium |
| Local Learning FS (LLFS) | Local structure | Low |
| Locality Preserving Projection FS | LPP-based | Low |
| Neighborhood Preserving FS | Neighborhood structure | Low |
| Sparse Spectral FS | Sparse constraints | Low |

### 4.2 Graph-Based Methods
| Method | Description | Priority |
|--------|-------------|----------|
| PageRank FS | PageRank on feature graph | Medium |
| HITS FS | Hub/authority scores | Low |
| Graph Centrality FS | Centrality measures | Medium |
| Betweenness Centrality FS | Path-based | Low |
| Closeness Centrality FS | Distance-based | Low |
| Eigenvector Centrality FS | Eigenvector-based | Low |
| Katz Centrality FS | Weighted paths | Low |
| Community Detection FS | Graph clustering | Low |
| Minimum Spanning Tree FS | MST-based | Low |
| Graph Cut FS | Cut-based | Low |
| Markov Blanket | Causal graph | High |
| Bayesian Network FS | DAG structure | Medium |
| Constraint-Based FS | PC algorithm | Medium |
| FCI (Fast Causal Inference) | Latent variables | Low |
| GES (Greedy Equivalence Search) | Score-based causal | Low |

---

## 5. FILTER METHODS - STATISTICAL/RANKING

### 5.1 Variance-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Variance Threshold | ✅ Already implemented | Done |
| Coefficient of Variation | Normalized variance | Medium |
| Mean Absolute Deviation | Robust dispersion | Medium |
| Median Absolute Deviation | Very robust dispersion | High |
| Interquartile Range | IQR-based | Medium |
| Range Ratio | Max-min ratio | Low |
| Signal-to-Noise Ratio | SNR ranking | Medium |
| Dispersion Ratio | Between/within dispersion | Low |

### 5.2 Distribution-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Skewness Filter | Asymmetry filter | Medium |
| Kurtosis Filter | Tail heaviness | Medium |
| Entropy Filter | Distribution complexity | Medium |
| Bimodality Coefficient | Multi-modal detection | Low |
| Dip Test | Unimodality test | Low |
| Hartigan's Dip | Dip statistic | Low |
| Outlier Proportion | Outlier-based | Medium |
| Missing Value Ratio | Completeness | High |
| Unique Value Ratio | Cardinality | Medium |
| Zero Variance | Constant detection | High |

---

## 6. WRAPPER METHODS - SEQUENTIAL

| Method | Description | Priority |
|--------|-------------|----------|
| Forward Selection | ✅ Already implemented | Done |
| Backward Elimination | ✅ Already implemented | Done |
| Stepwise Selection | Forward + backward combined | High |
| SFFS (Sequential Floating Forward) | Forward with backtracking | **Critical** |
| SFBS (Sequential Floating Backward) | Backward with backtracking | High |
| Plus-L Minus-R | Add L, remove R | Medium |
| Generalized Sequential | Parameterized sequential | Low |
| Adaptive Sequential | Adaptive step size | Low |
| Bidirectional Search | Both directions | Medium |
| Beam Search | Multiple paths | Medium |
| Best-First Search | Priority queue | Medium |
| Branch and Bound | Optimal search | Medium |
| A* Search | Heuristic search | Low |
| IDA* (Iterative Deepening A*) | Memory-efficient A* | Low |
| LRS (Linear Reference Search) | Linear approximation | Low |

---

## 7. WRAPPER METHODS - EXHAUSTIVE/COMPLETE

| Method | Description | Priority |
|--------|-------------|----------|
| Exhaustive Search | All 2^n subsets | Medium |
| Approximate Exhaustive | Sampled subsets | Medium |
| Random Subset | Random sampling | Medium |
| Latin Hypercube Sampling | Stratified random | Low |
| Sobol Sequence | Quasi-random | Low |
| Halton Sequence | Low-discrepancy | Low |
| Orthogonal Arrays | Experimental design | Low |
| Factorial Design | Full/fractional factorial | Low |
| D-Optimal Design | Optimal experimental | Low |
| Bootstrap Aggregated | Bagged subsets | Medium |

---

## 8. WRAPPER METHODS - METAHEURISTIC/EVOLUTIONARY

### 8.1 Genetic/Evolutionary
| Method | Description | Priority |
|--------|-------------|----------|
| Genetic Algorithm (GA) | Classic evolutionary | **Critical** |
| Steady-State GA | Incremental replacement | Medium |
| Micro-GA | Small population GA | Low |
| CHC (Cross-generational Heterogeneous) | Elitist GA | Low |
| Scatter Search | Population-based | Medium |
| Memetic Algorithm | GA + local search | Medium |
| Evolution Strategy (ES) | Self-adaptive | Medium |
| CMA-ES | Covariance matrix adaptation | Medium |
| Differential Evolution (DE) | Vector differences | High |
| JADE | Adaptive DE | Medium |
| SHADE | Success-history DE | Low |
| L-SHADE | Linear population DE | Low |
| Evolutionary Programming | No crossover | Low |
| Genetic Programming | Tree-based | Low |
| Gene Expression Programming | Linear + tree | Low |
| Grammatical Evolution | Grammar-based | Low |
| Estimation of Distribution (EDA) | Probabilistic models | Medium |
| PBIL (Population-Based Incremental) | Probability vector | Medium |
| UMDA (Univariate Marginal DA) | Univariate EDA | Medium |
| BOA (Bayesian Optimization Algorithm) | Bayesian network EDA | Low |
| ECGA (Extended Compact GA) | Linkage learning | Low |
| hBOA (Hierarchical BOA) | Hierarchical Bayesian | Low |
| DSMGA (Dependency Structure Matrix GA) | DSM-based | Low |
| LTGA (Linkage Tree GA) | Tree-based linkage | Low |

### 8.2 Swarm Intelligence
| Method | Description | Priority |
|--------|-------------|----------|
| Particle Swarm Optimization (PSO) | Swarm-based | High |
| Binary PSO | Discrete PSO | High |
| Discrete PSO | Integer PSO | Medium |
| Quantum PSO | Quantum-inspired | Low |
| Adaptive PSO | Self-adaptive | Medium |
| Comprehensive Learning PSO | Multi-swarm | Low |
| Unified PSO | Combined strategies | Low |
| Firefly Algorithm | Light attraction | Medium |
| Glowworm Swarm | Local attraction | Low |
| Ant Colony Optimization (ACO) | Pheromone trails | High |
| Ant System | Original ACO | Medium |
| Max-Min Ant System | Bounded pheromones | Medium |
| Ant Colony System | Enhanced ACO | Medium |
| Bee Algorithm | Foraging behavior | Medium |
| Artificial Bee Colony (ABC) | Employed/onlooker/scout | High |
| Bees Algorithm | Neighborhood search | Medium |
| Bacterial Foraging | Chemotaxis | Low |
| Cat Swarm Optimization | Seeking/tracing | Low |
| Fish School Search | Fish schooling | Low |
| Krill Herd | Krill movement | Low |
| Grey Wolf Optimizer (GWO) | Alpha/beta/delta/omega | High |
| Whale Optimization (WOA) | Bubble-net hunting | High |
| Dragonfly Algorithm | Static/dynamic swarm | Medium |
| Grasshopper Optimization | Swarm attraction/repulsion | Medium |
| Moth-Flame Optimization | Light navigation | Medium |
| Salp Swarm Algorithm | Chain movement | Low |
| Harris Hawks Optimization | Surprise pounce | Medium |
| Marine Predators Algorithm | Predator-prey | Low |
| Slime Mould Algorithm | Oscillation | Low |

### 8.3 Physics-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Simulated Annealing (SA) | Thermal annealing | High |
| Parallel Tempering | Multiple temperatures | Low |
| Quantum Annealing | Quantum tunneling | Low |
| Gravitational Search (GSA) | Newton's gravity | Medium |
| Binary GSA | Discrete GSA | Medium |
| Central Force Optimization | Gravity/probes | Low |
| Black Hole Algorithm | Event horizon | Low |
| Big Bang-Big Crunch | Cosmology | Low |
| Charged System Search | Coulomb's law | Low |
| Electromagnetic Algorithm | EM forces | Low |
| Water Cycle Algorithm | Evaporation/rain | Low |
| Water Wave Optimization | Wave propagation | Low |
| Wind Driven Optimization | Air pressure | Low |

### 8.4 Human/Social-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Harmony Search | Musical improvisation | Medium |
| Improved Harmony Search | Enhanced version | Medium |
| Teaching-Learning Optimization (TLBO) | Teacher/learner | High |
| Brain Storm Optimization | Brainstorming | Low |
| Group Search Optimizer | Producer/scrounger | Low |
| Social Learning Optimization | Imitation | Low |
| League Championship Algorithm | Sports league | Low |
| Imperialist Competitive | Colonial competition | Low |
| Cultural Algorithm | Belief space | Low |

### 8.5 Other Nature-Inspired
| Method | Description | Priority |
|--------|-------------|----------|
| Cuckoo Search | Brood parasitism | High |
| Bat Algorithm | Echolocation | Medium |
| Flower Pollination | Pollination | Medium |
| Invasive Weed Optimization | Weed colonization | Low |
| Forest Optimization | Tree seeding | Low |
| Tree-Seed Algorithm | Seed dispersal | Low |
| Symbiotic Organisms Search | Mutualism/parasitism | Low |
| Coral Reef Optimization | Coral reproduction | Low |

---

## 9. WRAPPER METHODS - HYBRID SEARCH

| Method | Description | Priority |
|--------|-------------|----------|
| Tabu Search | Memory-based | High |
| Reactive Tabu Search | Adaptive tabu | Medium |
| Variable Neighborhood Search (VNS) | Neighborhood change | High |
| Basic VNS | Shaking + local | Medium |
| General VNS | Multiple descents | Medium |
| Iterated Local Search (ILS) | Perturbation + local | High |
| Guided Local Search (GLS) | Penalty-guided | Medium |
| Large Neighborhood Search (LNS) | Destroy/repair | Medium |
| Adaptive LNS | Adaptive operators | Low |
| Path Relinking | Trajectory between solutions | Medium |
| GRASP | Greedy + local search | High |
| Reactive GRASP | Adaptive alpha | Medium |
| Pilot Method | Look-ahead | Low |
| Extremal Optimization | Bak-Sneppen model | Low |

---

## 10. WRAPPER METHODS - BAYESIAN/PROBABILISTIC

| Method | Description | Priority |
|--------|-------------|----------|
| Bayesian Optimization | Surrogate model | High |
| Gaussian Process FS | GP-based | Medium |
| Tree-Parzen Estimator (TPE) | Tree-structured | High |
| Sequential Model-Based (SMAC) | Random forest surrogate | Medium |
| Hyperband | Bandit-based | Medium |
| BOHB (Bayesian Optimization + Hyperband) | Combined | Medium |
| Expected Improvement | Acquisition function | Medium |
| Probability of Improvement | Risk-averse | Low |
| Upper Confidence Bound | Exploration bonus | Medium |
| Knowledge Gradient | One-step lookahead | Low |
| Entropy Search | Information gain | Low |
| Thompson Sampling | Posterior sampling | Medium |
| Multi-Armed Bandit FS | Bandit formulation | Medium |
| UCB1 | Upper confidence bound | Medium |
| Successive Halving | Resource allocation | Medium |
| Monte Carlo Tree Search | Tree search + simulation | Low |

---

## 11. EMBEDDED METHODS - REGULARIZATION

### 11.1 L1-Based (Sparsity)
| Method | Description | Priority |
|--------|-------------|----------|
| Lasso (L1) | Basic L1 penalty | **Critical** |
| Adaptive Lasso | Weighted L1 | High |
| Group Lasso | L1 on groups | High |
| Sparse Group Lasso | Group + within-group L1 | Medium |
| Fused Lasso | L1 + adjacent differences | Medium |
| Graphical Lasso | Sparse precision matrix | Medium |
| Elastic Net | L1 + L2 | **Critical** |
| Adaptive Elastic Net | Weighted elastic net | Medium |
| Exclusive Lasso | Exclusive group sparsity | Low |
| Dantzig Selector | Constrained L1 | Medium |
| SQRT-Lasso | Scale-free | Low |
| Scaled Lasso | Self-tuning | Low |

### 11.2 Non-Convex Penalties
| Method | Description | Priority |
|--------|-------------|----------|
| SCAD | Smoothly clipped absolute | High |
| MCP (Minimax Concave) | Concave penalty | High |
| Log Penalty | Logarithmic | Low |
| Capped L1 | Truncated | Low |
| Bridge Regression | L_q penalty | Medium |

### 11.3 Structured Sparsity
| Method | Description | Priority |
|--------|-------------|----------|
| Tree Lasso | Tree structure | Low |
| Hierarchical Lasso | Hierarchy | Low |
| Overlapping Group Lasso | Overlapping groups | Low |
| Network Lasso | Network penalties | Low |
| Low-Rank + Sparse | Decomposition | Low |
| Nuclear Norm | Low-rank | Low |

---

## 12. EMBEDDED METHODS - TREE/ENSEMBLE IMPORTANCE

| Method | Description | Priority |
|--------|-------------|----------|
| Decision Tree Importance | Node impurity | High |
| Gini Importance | Gini decrease | High |
| Entropy Importance | IG decrease | High |
| Mean Decrease Impurity (MDI) | Averaged impurity | High |
| Mean Decrease Accuracy (MDA) | Permutation-based | High |
| Random Forest Importance | RF importance | High |
| Extra Trees Importance | Extremely randomized | Medium |
| Gradient Boosting Importance | Boosting importance | High |
| XGBoost Importance | XGB scores | Medium |
| LightGBM Importance | LGBM scores | Medium |
| CatBoost Importance | CatBoost scores | Low |
| AdaBoost Importance | AdaBoost weights | Medium |
| Gain Importance | Gain-based | Medium |
| Cover Importance | Coverage | Low |
| Frequency Importance | Split frequency | Medium |
| Permutation Importance | Shuffle-based | **Critical** |
| Drop-Column Importance | Leave-one-out | High |
| Boruta | Shadow features | **Critical** |

---

## 13. EMBEDDED METHODS - NEURAL NETWORK

### 13.1 Weight/Gradient-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Weight Magnitude | Absolute weights | High |
| Gradient Magnitude | Gradient norms | High |
| Fisher Information | Information matrix | Medium |
| Layer-wise Relevance Propagation (LRP) | Backprop attribution | High |
| DeepLIFT | Reference-based | High |
| DeepSHAP | SHAP + DeepLIFT | High |
| Integrated Gradients | Path integration | High |
| SmoothGrad | Noisy gradients | Medium |
| GradCAM | Class activation | High |
| GradCAM++ | Improved GradCAM | Medium |
| Score-CAM | Gradient-free CAM | Medium |

### 13.2 Attention-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Attention Weights | Transformer attention | High |
| Self-Attention Scores | Self-attention | High |
| Multi-Head Average | Averaged heads | Medium |
| Attention Rollout | Propagated attention | Medium |
| Attention Flow | Graph-based flow | Low |

### 13.3 Pruning-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Magnitude Pruning | Weight magnitude | High |
| Gradient Pruning | Gradient magnitude | Medium |
| Sensitivity Analysis | Remove and observe | High |
| Lottery Ticket | Sparse subnetworks | Medium |
| SNIP | Single-shot pruning | Medium |
| Structured Pruning | Groups/channels | Medium |

### 13.4 Input-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Input Gradient | ∂L/∂x | High |
| Input × Gradient | Element-wise product | High |
| Guided Backprop | Positive gradients | Medium |
| Occlusion Sensitivity | Sliding window | High |
| RISE | Randomized input sampling | Medium |
| SHAP (KernelExplainer) | Model-agnostic SHAP | **Critical** |
| LIME | Local linear | **Critical** |
| Anchors | Rule-based | Medium |
| Counterfactual Explanations | Minimal changes | Medium |

---

## 14. MODEL-AGNOSTIC METHODS

| Method | Description | Priority |
|--------|-------------|----------|
| Permutation Importance | Shuffle features | **Critical** |
| Drop-Column Importance | Retrain without | High |
| SHAP (Shapley Values) | Game theory | **Critical** |
| TreeSHAP | Tree-specific SHAP | High |
| KernelSHAP | Kernel-based SHAP | High |
| LIME | Local surrogate | **Critical** |
| Partial Dependence | PD plots | High |
| Individual Conditional Expectation | ICE plots | Medium |
| Accumulated Local Effects | ALE plots | Medium |
| Feature Interaction | H-statistic | Medium |
| Friedman's H-statistic | Interaction strength | Medium |
| Model Class Reliance | MCR | Low |
| Conditional Permutation | Conditional shuffle | Medium |
| Model-X Knockoffs | Knockoff filter | High |
| Deep Knockoffs | Neural knockoffs | Low |
| LOCO (Leave-One-Covariate-Out) | Leave-out | Medium |
| SAGE | Shapley additive global | Medium |

---

## 15. UNSUPERVISED FEATURE SELECTION

### 15.1 Variance/Statistics-Based
| Method | Description | Priority |
|--------|-------------|----------|
| Variance Threshold | ✅ Already implemented | Done |
| Coefficient of Variation | Relative variability | Medium |
| MAD (Median Absolute Deviation) | Robust dispersion | High |
| IQR-Based Selection | Interquartile range | Medium |

### 15.2 Reconstruction-Based
| Method | Description | Priority |
|--------|-------------|----------|
| PCA Loadings | Principal components | High |
| Factor Analysis | Latent factors | Medium |
| ICA | Independent components | Medium |
| NMF (Non-negative Matrix) | Non-negative factors | Medium |
| Sparse PCA | Sparse loadings | High |
| Robust PCA | Outlier-robust | Medium |
| Autoencoder Selection | Reconstruction error | High |
| Sparse Autoencoder | Sparse representations | Medium |
| Variational Autoencoder | VAE importance | Medium |

### 15.3 Clustering-Based
| Method | Description | Priority |
|--------|-------------|----------|
| K-Means Feature Selection | Centroid-based | Medium |
| Hierarchical Clustering FS | Dendrogram-based | Medium |
| Spectral Clustering FS | Spectral features | Medium |
| DBSCAN-Based FS | Density-based | Low |

### 15.4 Graph-Based Unsupervised
| Method | Description | Priority |
|--------|-------------|----------|
| UDFS (Unsupervised Discriminative) | Discriminative unsupervised | Medium |
| NDFS (Nonnegative Discriminative) | Non-negative constraint | Low |
| RUFS (Robust Unsupervised) | Robust to noise | Medium |

---

## 16. SEMI-SUPERVISED FEATURE SELECTION

| Method | Description | Priority |
|--------|-------------|----------|
| Semi-Supervised LASSO | Labeled + unlabeled | Medium |
| Graph-Based SS FS | Graph with labels | Medium |
| Self-Training FS | Pseudo-labels | Medium |
| Co-Training FS | Multi-view | Low |
| Label Propagation FS | Label spreading | Medium |
| Manifold Regularization FS | Manifold constraint | Low |

---

## 17. MULTI-LABEL FEATURE SELECTION

| Method | Description | Priority |
|--------|-------------|----------|
| Multi-Label ReliefF | ML-ReliefF | High |
| Multi-Label Information Gain | ML-IG | High |
| Multi-Label Mutual Information | ML-MI | High |
| ML-KNN FS | k-NN based | Medium |
| PMU (Pairwise Mutual Information) | Pairwise MI | Medium |
| MDDM (Dependence Degree Max) | Dependence-based | Low |
| LLSF (Label-Specific Feature) | Label-specific | Low |

---

## 18. ONLINE/STREAMING FEATURE SELECTION

| Method | Description | Priority |
|--------|-------------|----------|
| Online Feature Selection | Sequential updates | High |
| Streaming Feature Selection | Data streams | High |
| OSFS (Online Streaming FS) | Streaming | Medium |
| Fast-OSFS | Fast streaming | Medium |
| SAOLA | Scalable and accurate | Medium |
| Alpha-Investing | Alpha control | Medium |
| Grafting | Online L1 | Low |
| Incremental Feature Selection | Incremental updates | Medium |
| Windowed Feature Selection | Sliding window | Medium |
| Concept Drift FS | Drift handling | Medium |

---

## 19. HIGH-DIMENSIONAL FEATURE SELECTION

| Method | Description | Priority |
|--------|-------------|----------|
| Sure Independence Screening (SIS) | Ultra-high dimensional | High |
| Iterative SIS (ISIS) | Iterative version | High |
| DC-SIS | Distance correlation SIS | Medium |
| Knockoff Filter | False discovery control | High |
| Model-X Knockoffs | Model-X framework | High |
| SLOPE | Sorted L-One | Medium |
| False Discovery Rate Control | FDR-based | High |
| Benjamini-Hochberg | FDR procedure | High |
| Bonferroni Correction | FWER control | High |
| Stability Selection | Subsampling | **Critical** |
| Complementary Pairs Stability | CPSS | Medium |
| Random Lasso | Randomized | Medium |
| Bolasso | Bootstrap Lasso | Medium |

---

## 20. ENSEMBLE FEATURE SELECTION

| Method | Description | Priority |
|--------|-------------|----------|
| Ensemble Feature Selection | Multiple methods | High |
| Feature Selection Aggregation | Aggregated rankings | High |
| Voting-Based FS | Majority vote | High |
| Weighted Voting FS | Weighted combination | Medium |
| Stacking-Based FS | Stacked selectors | Medium |
| Bootstrap Aggregating FS | Bagged selection | High |
| Random Subspace FS | Random features | Medium |
| Diversity-Based Ensemble | Diverse selectors | Low |
| Rank Aggregation | Rank combination | High |
| Borda Count | Ranked voting | Medium |
| Robust Rank Aggregation | Statistical aggregation | Medium |

---

## 21. MULTI-OBJECTIVE FEATURE SELECTION

| Method | Description | Priority |
|--------|-------------|----------|
| NSGA-II FS | Non-dominated sorting | High |
| NSGA-III FS | Many-objective | Medium |
| MOEA/D FS | Decomposition-based | Medium |
| SPEA2 FS | Strength Pareto | Medium |
| MOPSO FS | Multi-objective PSO | Medium |
| Pareto-Based FS | Pareto ranking | Medium |
| Weighted Sum FS | Scalarization | Medium |

---

## 22. DOMAIN-SPECIFIC METHODS

### 22.1 Bioinformatics/Genomics
| Method | Description | Priority |
|--------|-------------|----------|
| SAM (Significance Analysis of Microarrays) | Gene expression | Medium |
| LIMMA | Linear models | Medium |
| DESeq2 | RNA-seq | Low |
| Pathway-Based Selection | Biological pathways | Low |
| Network-Based Gene Selection | PPI networks | Low |

### 22.2 Text/NLP
| Method | Description | Priority |
|--------|-------------|----------|
| TF-IDF Selection | Term frequency | High |
| BM25 Selection | Okapi BM25 | Medium |
| Chi-Squared for Text | Document-term | High |
| Information Gain Text | IG for terms | High |
| Document Frequency | DF-based | Medium |

### 22.3 Time Series
| Method | Description | Priority |
|--------|-------------|----------|
| Autocorrelation-Based | ACF/PACF | High |
| Spectral Feature Selection | Frequency domain | Medium |
| Granger Causality | Causal relationships | High |
| Cross-Correlation Selection | Lagged correlation | Medium |
| Shapelet Selection | Shapelet discovery | Medium |

### 22.4 Image/Computer Vision
| Method | Description | Priority |
|--------|-------------|----------|
| CNN Feature Selection | Deep features | High |
| Channel Pruning | Channel selection | Medium |
| Attention-Based Image FS | Visual attention | Medium |
| Saliency-Based Selection | Saliency maps | Medium |

---

## 23. FAIRNESS-AWARE FEATURE SELECTION

| Method | Description | Priority |
|--------|-------------|----------|
| Fair Feature Selection | Bias removal | Medium |
| Disparate Impact Removal | Impact mitigation | Medium |
| Equalized Odds FS | Odds equalization | Low |
| Demographic Parity FS | Demographic fairness | Low |

---

## 24. PRIVACY-PRESERVING FEATURE SELECTION

| Method | Description | Priority |
|--------|-------------|----------|
| Differential Privacy FS | DP-based | Medium |
| Federated Feature Selection | Federated learning | Medium |
| Secure Multi-Party FS | MPC-based | Low |

---

## Priority Legend

- **Critical**: Must implement - industry standard, widely used
- **High**: Should implement - commonly requested
- **Medium**: Nice to have - specialized use cases
- **Low**: Future consideration - niche applications

## Implementation Order (Recommended)

### Phase 1: Critical Methods
1. mRMR (Min Redundancy Max Relevance)
2. ReliefF
3. Fisher Score
4. Boruta
5. Permutation Importance
6. SHAP-based Selection
7. LIME-based Selection
8. Stability Selection
9. Lasso-based Selection
10. Elastic Net Selection
11. SFFS (Sequential Floating Forward Selection)
12. Genetic Algorithm FS

### Phase 2: High Priority
13. Information Gain / Gain Ratio
14. Spearman/Kendall Correlation
15. RFECV (RFE with Cross-Validation)
16. PSO Feature Selection
17. Simulated Annealing FS
18. Laplacian Score
19. FCBF
20. CFS
21. Knockoff Filter
22. Ensemble Feature Selection

### Phase 3: Medium Priority
- Remaining methods based on user demand
