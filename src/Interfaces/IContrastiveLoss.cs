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
    /// <returns>
    /// A single-element tensor holding the loss, carrying tape history so it can be
    /// differentiated. Read element <c>[0]</c> for the scalar value.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This returns a <see cref="Tensor{T}"/> rather than a bare <c>T</c> on purpose. A scalar has
    /// no tape history, so an objective returning one can be MEASURED but never TRAINED — and every
    /// implementation of this interface previously returned a scalar assembled from host loops over
    /// tensor indexers. The entire family (InfoNCE, NT-Xent, BYOL, DINO, Barlow Twins, MAE) was
    /// therefore undifferentiable, which is why models reaching for a published contrastive
    /// objective silently fell back to a pointwise loss such as mean squared error instead.
    /// </para>
    /// <para>
    /// Implementations must build the result entirely from <c>IEngine</c> operations. Indexing a
    /// tensor to read values into host arithmetic severs the gradient, and it does so silently:
    /// the number returned still looks like a loss.
    /// </para>
    /// </remarks>
    Tensor<T> ComputeLoss(Tensor<T> view1, Tensor<T> view2);
}
