using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for contrastive and self-supervised loss functions that operate on pairs
/// of embeddings/representations rather than predictions vs ground truth labels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unlike <see cref="ILossFunction{T}"/> which compares predictions to actual labels,
/// contrastive losses compare two views/augmentations of the same data to learn
/// representations. Examples include SimCLR's NT-Xent, BYOL's regression loss,
/// and Barlow Twins' redundancy reduction loss.
/// </para>
/// </remarks>
public interface IContrastiveLoss<T>
{
    /// <summary>
    /// Computes the contrastive loss between two embedding tensors.
    /// </summary>
    /// <param name="view1">First view/augmentation embeddings.</param>
    /// <param name="view2">Second view/augmentation embeddings.</param>
    /// <returns>The scalar loss value.</returns>
    T ComputeLoss(Tensor<T> view1, Tensor<T> view2);
}
