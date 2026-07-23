namespace AiDotNet.Enums;

/// <summary>
/// Types of contrastive loss functions for knowledge distillation.
/// </summary>
public enum ContrastiveLossType
{
    /// <summary>
    /// InfoNCE (Noise Contrastive Estimation) loss.
    /// </summary>
    InfoNCE,

    /// <summary>
    /// Triplet loss with margin.
    /// </summary>
    TripletLoss,

    /// <summary>
    /// NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    /// </summary>
    NTXent
}
