namespace AiDotNet.Enums;

/// <summary>
/// Specifies the algorithm to use for causal structure learning (DAG discovery).
/// </summary>
/// <remarks>
/// <para>
/// Causal discovery algorithms learn the causal structure (a Directed Acyclic Graph or DAG)
/// from observational data. Different algorithms make different assumptions about the data
/// (linearity, Gaussianity, faithfulness) and use different strategies (constraint testing,
/// score optimization, continuous optimization).
/// </para>
/// <para>
/// <b>For Beginners:</b> These algorithms figure out which variables cause which other variables
/// by analyzing patterns in your data. Think of it like a detective figuring out cause-and-effect
/// relationships. Different algorithms are like different detective methods — some test independence
/// relationships, some optimize a score, and some use advanced math to find the best graph.
/// </para>
/// </remarks>
public enum CausalDiscoveryAlgorithmType
{
    // ──────────────────────────────────────────────────────
    // Category 1: Continuous Optimization — NOTEARS Family
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// NOTEARS Linear — continuous optimization with tr(e^(W∘W))-d acyclicity constraint.
    /// </summary>
    /// <remarks>
    /// <para>Uses augmented Lagrangian with L-BFGS-B inner solver. Assumes linear relationships.</para>
    /// <para>Reference: Zheng et al. (2018), "DAGs with NO TEARS: Continuous Optimization for Structure Learning"</para>
    /// </remarks>
    NOTEARSLinear,

    /// <summary>
    /// NOTEARS Nonlinear — extends NOTEARS with MLP (multi-layer perceptron) for nonlinear relationships.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Zheng et al. (2020), "Learning Sparse Nonparametric DAGs"</para>
    /// </remarks>
    NOTEARSNonlinear,

    /// <summary>
    /// NOTEARS with Sobolev basis functions for nonlinear relationships.
    /// </summary>
    NOTEARSSobolev,

    /// <summary>
    /// NOTEARS Low-Rank — low-rank approximation for scalability to high dimensions.
    /// </summary>
    NOTEARSLowRank,

    /// <summary>
    /// DAGMA Linear — log-determinant acyclicity constraint via M-matrices. ~10x faster than NOTEARS.
    /// </summary>
    /// <remarks>
    /// <para>Uses central path barrier method instead of augmented Lagrangian.</para>
    /// <para>Reference: Bello et al. (2022), "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization"</para>
    /// </remarks>
    DAGMALinear,

    /// <summary>
    /// DAGMA Nonlinear — extends DAGMA with neural network function approximation.
    /// </summary>
    DAGMANonlinear,

    /// <summary>
    /// GOLEM — likelihood-based single-loop optimization without augmented Lagrangian.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Ng et al. (2020), "On the Role of Sparsity and DAG Constraints for Learning Linear DAGs"</para>
    /// </remarks>
    GOLEM,

    /// <summary>
    /// NoCurl — curl-free constraint for acyclicity.
    /// </summary>
    NoCurl,

    /// <summary>
    /// MCSL — Multi-scale Causal Structure Learning.
    /// </summary>
    MCSL,

    /// <summary>
    /// CORL — Causal Order learning via Reinforcement Learning.
    /// </summary>
    CORL,

    // ──────────────────────────────────────────────────────
    // Category 2: Score-Based Search
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// GES — Greedy Equivalence Search. Searches over DAG equivalence classes using BIC scoring.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Chickering (2002), "Optimal Structure Identification With Greedy Search"</para>
    /// </remarks>
    GES,

    /// <summary>
    /// FGES — Fast Greedy Equivalence Search. Parallelized version of GES for large datasets.
    /// </summary>
    FGES,

    /// <summary>
    /// Hill Climbing — greedy local search with BIC or BDeu scoring.
    /// </summary>
    HillClimbing,

    /// <summary>
    /// Tabu Search — hill climbing with a tabu list to escape local optima.
    /// </summary>
    TabuSearch,

    /// <summary>
    /// K2 Algorithm — score-based search with a known variable ordering.
    /// </summary>
    K2,

    /// <summary>
    /// GRaSP — Greedy Relaxation of the Sparsest Permutation.
    /// </summary>
    GRaSP,

    /// <summary>
    /// BOSS — Bayesian Optimal Structure Search.
    /// </summary>
    BOSS,

    /// <summary>
    /// Exact Search — dynamic programming for exact structure learning (exponential complexity).
    /// </summary>
    ExactSearch,

    // ──────────────────────────────────────────────────────
    // Category 3: Constraint-Based
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// PC Algorithm — the gold standard constraint-based method using conditional independence tests.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Spirtes et al. (2000), "Causation, Prediction, and Search"</para>
    /// </remarks>
    PC,

    /// <summary>
    /// FCI — Fast Causal Inference. Handles latent (unobserved) confounders.
    /// </summary>
    FCI,

    /// <summary>
    /// RFCI — Really Fast Causal Inference. Faster approximation of FCI.
    /// </summary>
    RFCI,

    /// <summary>
    /// MMPC — Max-Min Parents and Children. Finds local causal neighborhood.
    /// </summary>
    MMPC,

    /// <summary>
    /// CPC — Conservative PC. More conservative edge orientation than standard PC.
    /// </summary>
    CPC,

    /// <summary>
    /// CD-NOD — Causal Discovery from Nonstationary/heterogeneous Data.
    /// </summary>
    CDNOD,

    /// <summary>
    /// IAMB — Incremental Association Markov Blanket.
    /// </summary>
    IAMB,

    /// <summary>
    /// Fast-IAMB — Faster variant of IAMB.
    /// </summary>
    FastIAMB,

    /// <summary>
    /// Markov Blanket discovery via the Grow-Shrink algorithm.
    /// </summary>
    MarkovBlanket,

    // ──────────────────────────────────────────────────────
    // Category 4: Hybrid
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// MMHC — Max-Min Hill Climbing. Combines MMPC constraint phase with Hill Climbing score phase.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Tsamardinos et al. (2006)</para>
    /// </remarks>
    MMHC,

    /// <summary>
    /// H2PC — Hybrid HPC algorithm.
    /// </summary>
    H2PC,

    /// <summary>
    /// GFCI — Greedy FCI. Hybrid of GES and FCI for latent confounders.
    /// </summary>
    GFCI,

    /// <summary>
    /// PC-NOTEARS — Hybrid combining PC skeleton with NOTEARS optimization.
    /// </summary>
    PCNOTEARS,

    /// <summary>
    /// RSMAX2 — Restricted maximization algorithm.
    /// </summary>
    RSMAX2,

    // ──────────────────────────────────────────────────────
    // Category 5: Functional / ICA-Based
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// ICA-LiNGAM — Linear Non-Gaussian Acyclic Model using Independent Component Analysis.
    /// </summary>
    /// <remarks>
    /// <para>Assumes linear relationships with non-Gaussian noise. Recovers unique DAG.</para>
    /// <para>Reference: Shimizu et al. (2006)</para>
    /// </remarks>
    ICALiNGAM,

    /// <summary>
    /// DirectLiNGAM — Direct method for LiNGAM without ICA. Uses iterative regression and independence testing.
    /// </summary>
    DirectLiNGAM,

    /// <summary>
    /// VAR-LiNGAM — LiNGAM for time series via Vector Autoregressive model.
    /// </summary>
    VARLiNGAM,

    /// <summary>
    /// RCD — Repetitive Causal Discovery. LiNGAM variant handling latent confounders.
    /// </summary>
    RCD,

    /// <summary>
    /// CAM-UV — Causal Additive Models with Unobserved Variables.
    /// </summary>
    CAMUV,

    /// <summary>
    /// ANM — Additive Noise Model. Bivariate causal discovery using regression residual independence.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Hoyer et al. (2008)</para>
    /// </remarks>
    ANM,

    /// <summary>
    /// PNL — Post-Nonlinear causal model.
    /// </summary>
    PNL,

    /// <summary>
    /// IGCI — Information-Geometric Causal Inference.
    /// </summary>
    IGCI,

    /// <summary>
    /// CAM — Causal Additive Models.
    /// </summary>
    CAM,

    /// <summary>
    /// CCDr — Concave penalized Coordinate Descent with reparameterization.
    /// </summary>
    CCDr,

    // ──────────────────────────────────────────────────────
    // Category 6: Time Series Causal Discovery
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// Granger Causality — tests whether one time series helps predict another.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Granger (1969)</para>
    /// </remarks>
    GrangerCausality,

    /// <summary>
    /// PCMCI — PC algorithm adapted for time series with momentary conditional independence.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Runge et al. (2019)</para>
    /// </remarks>
    PCMCI,

    /// <summary>
    /// PCMCI+ — Extension of PCMCI that also discovers contemporaneous causal links.
    /// </summary>
    PCMCIPlus,

    /// <summary>
    /// DYNOTEARS — Dynamic NOTEARS for time series structure learning.
    /// </summary>
    /// <remarks>
    /// <para>Reference: Pamfil et al. (2020)</para>
    /// </remarks>
    DYNOTEARS,

    /// <summary>
    /// TiMINo — Time series Model with Independent Noise.
    /// </summary>
    TiMINo,

    /// <summary>
    /// tsFCI — Time series Fast Causal Inference.
    /// </summary>
    TSFCI,

    /// <summary>
    /// LPCMCI — Latent PCMCI for time series with latent confounders.
    /// </summary>
    LPCMCI,

    /// <summary>
    /// NTS-NOTEARS — Non-stationary Time Series NOTEARS.
    /// </summary>
    NTSNOTEARS,

    /// <summary>
    /// CCM — Convergent Cross-Mapping for detecting causality in dynamical systems.
    /// </summary>
    CCM,

    /// <summary>
    /// Neural Granger Causality — deep learning extension of Granger causality.
    /// </summary>
    NeuralGranger,

    // ──────────────────────────────────────────────────────
    // Category 7: Deep Learning / Neural Network-Based
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// DAG-GNN — Graph Neural Network for DAG structure learning.
    /// </summary>
    DAGGNN,

    /// <summary>
    /// GraNDAG — Gradient-based Neural DAG Learning.
    /// </summary>
    GraNDAG,

    /// <summary>
    /// CASTLE — Causal Structure Learning.
    /// </summary>
    CASTLE,

    /// <summary>
    /// DECI — Deep End-to-end Causal Inference.
    /// </summary>
    DECI,

    /// <summary>
    /// GAE — Graph Autoencoder for structure learning.
    /// </summary>
    GAE,

    /// <summary>
    /// CGNN — Causal Generative Neural Networks.
    /// </summary>
    CGNN,

    /// <summary>
    /// TCDF — Temporal Causal Discovery Framework.
    /// </summary>
    TCDF,

    /// <summary>
    /// Amortized Causal Discovery — meta-learning approach to causal discovery.
    /// </summary>
    AmortizedCD,

    /// <summary>
    /// AVICI — Amortized Variational Inference for Causal Discovery.
    /// </summary>
    AVICI,

    /// <summary>
    /// CausalVAE — Variational Autoencoder for causal representation learning.
    /// </summary>
    CausalVAE,

    // ──────────────────────────────────────────────────────
    // Category 8: Bayesian
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// Order MCMC — MCMC over topological orderings for Bayesian structure learning.
    /// </summary>
    OrderMCMC,

    /// <summary>
    /// DiBS — Differentiable Bayesian Structure Learning.
    /// </summary>
    DiBS,

    /// <summary>
    /// BCD-Nets — Scalable variational Bayesian Causal Discovery.
    /// </summary>
    BCDNets,

    /// <summary>
    /// BayesDAG — Bayesian DAG learning with direct parameterization.
    /// </summary>
    BayesDAG,

    /// <summary>
    /// Partition MCMC — MCMC over DAG partitions.
    /// </summary>
    PartitionMCMC,

    /// <summary>
    /// Iterative MCMC — Iterative Bayesian structure learning.
    /// </summary>
    IterativeMCMC,

    // ──────────────────────────────────────────────────────
    // Category 9: Information-Theoretic
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// oCSE — Optimal Causation Entropy for detecting causal relationships.
    /// </summary>
    OCSE,

    /// <summary>
    /// Transfer Entropy — information-theoretic measure of directed information flow.
    /// </summary>
    TransferEntropy,

    /// <summary>
    /// Kraskov Mutual Information — k-nearest neighbor mutual information estimator.
    /// </summary>
    KraskovMI,

    // ──────────────────────────────────────────────────────
    // Category 10: Specialized
    // ──────────────────────────────────────────────────────

    /// <summary>
    /// GOBNILP — Integer Linear Programming for exact Bayesian network structure learning.
    /// </summary>
    GOBNILP
}
