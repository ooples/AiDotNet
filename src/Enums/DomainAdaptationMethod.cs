namespace AiDotNet.Enums;

/// <summary>
/// Specifies the method to use for domain adaptation in transfer learning.
/// </summary>
public enum DomainAdaptationMethod
{
    /// <summary>
    /// Maximum Mean Discrepancy - minimizes the distance between source and target distributions.
    /// </summary>
    MMD,
    
    /// <summary>
    /// Adversarial domain adaptation using a domain discriminator.
    /// </summary>
    Adversarial,
    
    /// <summary>
    /// Correlation Alignment - aligns second-order statistics of source and target.
    /// </summary>
    CORAL,
    
    /// <summary>
    /// Deep CORAL - extends CORAL to deep neural networks.
    /// </summary>
    DeepCORAL,
    
    /// <summary>
    /// Domain-Adversarial Neural Networks (DANN) with gradient reversal.
    /// </summary>
    GradientReversal,
    
    /// <summary>
    /// Joint Distribution Adaptation - aligns both marginal and conditional distributions.
    /// </summary>
    JDA,
    
    /// <summary>
    /// Balanced Distribution Adaptation.
    /// </summary>
    BDA,
    
    /// <summary>
    /// Optimal Transport for domain adaptation.
    /// </summary>
    OptimalTransport,
    
    /// <summary>
    /// Wasserstein Distance Guided Representation Learning.
    /// </summary>
    Wasserstein,
    
    /// <summary>
    /// Subspace alignment between source and target domains.
    /// </summary>
    SubspaceAlignment
}