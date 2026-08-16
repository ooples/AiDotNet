using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// Base class for the self-supervised and contrastive objectives.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Supplies the tensor engine the same way every other base in the library does
/// (<c>LossFunctionBase</c>, <c>ActivationFunctionBase</c>, <c>AdversarialAttackBase</c>), so an
/// objective builds its loss from <see cref="IEngine"/> operations and the result carries tape
/// history.
/// </para>
/// <para>
/// That matters more here than it looks. These objectives previously assembled their loss from
/// host loops over tensor indexers and returned a bare scalar, which cannot be differentiated —
/// so the whole family could be measured but never trained on, and models wanting a published
/// contrastive objective quietly fell back to a pointwise loss instead.
/// </para>
/// </remarks>
public abstract class ContrastiveLossBase<T> : IContrastiveLoss<T>
{
    /// <summary>
    /// Numeric operations for <typeparamref name="T"/>.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The tensor engine. Build every part of a loss from this so gradients survive.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <inheritdoc/>
    public abstract Tensor<T> ComputeLoss(Tensor<T> view1, Tensor<T> view2);
}
