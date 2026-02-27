namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies which federated aggregation strategy to use.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In federated learning, each client trains locally and sends an update.
/// The server then combines those updates using an aggregation strategy.
/// </remarks>
public enum FederatedAggregationStrategy
{
    /// <summary>
    /// Federated Averaging (FedAvg).
    /// </summary>
    FedAvg = 0,

    /// <summary>
    /// Federated Proximal (FedProx) for heterogeneity.
    /// </summary>
    FedProx = 1,

    /// <summary>
    /// Federated Batch Normalization (FedBN).
    /// </summary>
    FedBN = 2,

    /// <summary>
    /// Coordinate-wise median aggregation.
    /// </summary>
    Median = 3,

    /// <summary>
    /// Coordinate-wise trimmed mean aggregation.
    /// </summary>
    TrimmedMean = 4,

    /// <summary>
    /// Coordinate-wise winsorized mean aggregation.
    /// </summary>
    WinsorizedMean = 5,

    /// <summary>
    /// Robust Federated Aggregation (geometric median / RFA).
    /// </summary>
    Rfa = 6,

    /// <summary>
    /// Krum (Byzantine-robust selection).
    /// </summary>
    Krum = 7,

    /// <summary>
    /// Multi-Krum (select m central updates, then average).
    /// </summary>
    MultiKrum = 8,

    /// <summary>
    /// Bulyan (Multi-Krum selection + trimming).
    /// </summary>
    Bulyan = 9,

    /// <summary>
    /// MOON — Model-Contrastive learning. Uses contrastive loss between local/global representations
    /// to correct local drift under non-IID data. (Li et al., CVPR 2021)
    /// </summary>
    Moon = 10,

    /// <summary>
    /// FedNTD — Not-True Distillation. Self-distillation that only penalizes non-true class logits,
    /// preserving local knowledge while aligning with global model. (Lee et al., NeurIPS 2022)
    /// </summary>
    FedNtd = 11,

    /// <summary>
    /// FedLC — Logit Calibration. Adjusts local logits by class frequency to reduce bias
    /// caused by imbalanced label distributions. (Zhang et al., ICML 2022)
    /// </summary>
    FedLc = 12,

    /// <summary>
    /// FedDecorr — Decorrelation regularizer that encourages diverse feature representations
    /// across clients to reduce dimensional collapse. (Shi et al., ICML 2023)
    /// </summary>
    FedDecorr = 13,

    /// <summary>
    /// FedAlign — Feature alignment across clients using shared anchor representations.
    /// (Mendieta et al., CVPR 2022)
    /// </summary>
    FedAlign = 14,

    /// <summary>
    /// FedSAM — Sharpness-Aware Minimization for FL. Seeks flat minima for better
    /// generalization across heterogeneous client data. (Caldarola et al., 2022)
    /// </summary>
    FedSam = 15,

    /// <summary>
    /// FedMA — Matched Averaging. Uses Bayesian layer matching to align neuron permutations
    /// before averaging, solving the permutation invariance problem. (Wang et al., ICLR 2020)
    /// </summary>
    FedMa = 16,

    /// <summary>
    /// FedAA — Adaptive Aggregation. Learns optimal per-client aggregation weights using
    /// attention mechanism over gradient similarities. (2024)
    /// </summary>
    FedAa = 17,

    /// <summary>
    /// FLTrust — Server maintains a root dataset and computes trust scores by comparing
    /// client update directions. Only trusted updates are included. (Cao et al., NDSS 2021)
    /// </summary>
    FLTrust = 18,

    /// <summary>
    /// DnC — Divide and Conquer. Projects updates into random low-dimensional subspaces and
    /// applies spectral analysis to detect Byzantine outliers. (Shejwalkar and Houmansadr, USENIX 2021)
    /// </summary>
    DivideAndConquer = 19,

    /// <summary>
    /// Bucketing — Randomly partitions clients into buckets before applying a robust aggregation
    /// rule, provably improving the breakdown point. (Karimireddy et al., ICML 2022)
    /// </summary>
    Bucketing = 20,

    /// <summary>
    /// FLAME — Cosine-similarity filtering with adaptive clipping and DP noise injection
    /// for backdoor resistance. (Nguyen et al., USENIX 2022)
    /// </summary>
    Flame = 21,

    /// <summary>
    /// BOBA — Bayesian Optimal Byzantine-robust Aggregation with posterior inference
    /// over honest/malicious client labels. (2024)
    /// </summary>
    Boba = 22,

    /// <summary>
    /// OptiGradTrust — Optimized gradient trust scoring with EMA-based historical reputation
    /// tracking. Clients build trust over time via consistent aligned updates. (2024)
    /// </summary>
    OptiGradTrust = 23
}

