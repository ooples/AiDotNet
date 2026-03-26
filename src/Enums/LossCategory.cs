namespace AiDotNet.Enums;

/// <summary>
/// Categories of loss functions based on the type of learning task they serve.
/// </summary>
public enum LossCategory
{
    /// <summary>Classification losses (CrossEntropy, Focal, Hinge).</summary>
    Classification,
    /// <summary>Regression losses (MSE, MAE, Huber).</summary>
    Regression,
    /// <summary>Segmentation losses (Dice, Tversky, IoU).</summary>
    Segmentation,
    /// <summary>Ranking/metric learning losses (Triplet, Contrastive, MarginRanking).</summary>
    Ranking,
    /// <summary>Generative model losses (Adversarial, Perceptual, Reconstruction).</summary>
    Generation,
    /// <summary>Contrastive/self-supervised losses (InfoNCE, SimCLR, BYOL).</summary>
    Contrastive,
    /// <summary>Reconstruction losses (MSE, BCE for autoencoders/VAEs).</summary>
    Reconstruction,
    /// <summary>Adversarial losses (Wasserstein, Hinge for GANs).</summary>
    Adversarial,
    /// <summary>Regularization losses (L1, L2, KL divergence).</summary>
    Regularization,
    /// <summary>Physics-informed losses (PDE residual, boundary condition).</summary>
    PhysicsInformed
}
