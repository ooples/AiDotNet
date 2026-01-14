using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Provides stop-gradient operations for self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Stop-gradient (also called "detach" in PyTorch) prevents gradients
/// from flowing through a tensor during backpropagation. This is crucial for several SSL methods:</para>
///
/// <list type="bullet">
/// <item><b>SimSiam:</b> Stop-gradient on one branch prevents collapse without momentum encoder</item>
/// <item><b>BYOL:</b> Target network outputs are detached (no gradients to momentum encoder)</item>
/// <item><b>MoCo:</b> Memory bank entries and momentum encoder outputs are detached</item>
/// </list>
///
/// <para><b>Why stop-gradient?</b></para>
/// <para>Without stop-gradient, the model could "cheat" by making both branches output constants,
/// resulting in representation collapse. Stop-gradient forces asymmetry that prevents this.</para>
///
/// <para><b>Example usage:</b></para>
/// <code>
/// // SimSiam: asymmetric loss with stop-gradient
/// var z1 = encoder(x1);  // Online branch
/// var z2 = encoder(x2);  // Online branch
/// var p1 = predictor(z1);
/// var p2 = predictor(z2);
///
/// // Loss with stop-gradient - gradients only flow through predictor side
/// var loss = -CosineSimilarity(p1, StopGradient.Detach(z2)).Mean()
///          - CosineSimilarity(p2, StopGradient.Detach(z1)).Mean();
/// </code>
/// </remarks>
public static class StopGradient<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Detaches a tensor from the computation graph, preventing gradient flow.
    /// </summary>
    /// <param name="tensor">The tensor to detach.</param>
    /// <returns>A copy of the tensor that won't contribute to gradients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a copy of the tensor that acts as a "constant"
    /// during backpropagation. Gradients won't flow back through this tensor.</para>
    /// </remarks>
    public static Tensor<T> Detach(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        // Create a copy of the tensor data
        var dataCopy = new T[tensor.Length];
        Array.Copy(tensor.Data.ToArray(), dataCopy, tensor.Length);

        // Create new tensor with copied data (no gradient tracking)
        return new Tensor<T>(dataCopy, tensor.Shape);
    }

    /// <summary>
    /// Detaches a batch of tensors from the computation graph.
    /// </summary>
    /// <param name="tensors">The tensors to detach.</param>
    /// <returns>Copies of the tensors that won't contribute to gradients.</returns>
    public static Tensor<T>[] DetachBatch(params Tensor<T>[] tensors)
    {
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));

        var result = new Tensor<T>[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
        {
            result[i] = Detach(tensors[i]);
        }
        return result;
    }

    /// <summary>
    /// Applies stop-gradient to a vector.
    /// </summary>
    /// <param name="vector">The vector to detach.</param>
    /// <returns>A copy of the vector that won't contribute to gradients.</returns>
    public static Vector<T> Detach(Vector<T> vector)
    {
        if (vector is null) throw new ArgumentNullException(nameof(vector));

        var dataCopy = new T[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            dataCopy[i] = vector[i];
        }

        return new Vector<T>(dataCopy);
    }

    /// <summary>
    /// Creates a zero-gradient version of a tensor for the backward pass.
    /// </summary>
    /// <param name="tensor">The tensor shape to match.</param>
    /// <returns>A zero tensor with the same shape (for gradient accumulation).</returns>
    public static Tensor<T> ZeroGrad(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        var zeros = new T[tensor.Length];
        for (int i = 0; i < zeros.Length; i++)
        {
            zeros[i] = NumOps.Zero;
        }

        return new Tensor<T>(zeros, tensor.Shape);
    }

    /// <summary>
    /// Computes the symmetric loss with stop-gradient for SimSiam-style training.
    /// </summary>
    /// <param name="prediction1">Prediction from view 1 through predictor.</param>
    /// <param name="target1">Target from view 1 (will be detached).</param>
    /// <param name="prediction2">Prediction from view 2 through predictor.</param>
    /// <param name="target2">Target from view 2 (will be detached).</param>
    /// <param name="lossFunction">Function to compute loss between prediction and target.</param>
    /// <returns>The symmetric loss value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This implements the symmetric loss used in SimSiam:</para>
    /// <code>
    /// L = 0.5 * (loss(p1, stop_grad(z2)) + loss(p2, stop_grad(z1)))
    /// </code>
    /// </remarks>
    public static T SymmetricLoss(
        Tensor<T> prediction1,
        Tensor<T> target1,
        Tensor<T> prediction2,
        Tensor<T> target2,
        Func<Tensor<T>, Tensor<T>, T> lossFunction)
    {
        if (lossFunction is null) throw new ArgumentNullException(nameof(lossFunction));

        // Detach targets (stop-gradient)
        var detachedTarget2 = Detach(target2);
        var detachedTarget1 = Detach(target1);

        // Compute symmetric loss
        var loss1 = lossFunction(prediction1, detachedTarget2);
        var loss2 = lossFunction(prediction2, detachedTarget1);

        // Return average: 0.5 * (loss1 + loss2)
        return NumOps.Multiply(
            NumOps.FromDouble(0.5),
            NumOps.Add(loss1, loss2));
    }
}

/// <summary>
/// Marker interface for tensors that should not receive gradients.
/// </summary>
/// <remarks>
/// <para>This can be used to mark tensors at compile time as non-differentiable.
/// Useful for type-safe gradient handling in advanced scenarios.</para>
/// </remarks>
public interface IDetachedTensor<T>
{
    /// <summary>
    /// Gets the underlying tensor data.
    /// </summary>
    Tensor<T> Data { get; }
}

/// <summary>
/// A wrapper that marks a tensor as detached from the computation graph.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public readonly struct DetachedTensor<T> : IDetachedTensor<T>
{
    private readonly Tensor<T> _data;

    /// <summary>
    /// Initializes a new DetachedTensor wrapping the given tensor.
    /// </summary>
    /// <param name="tensor">The tensor to wrap (will be copied).</param>
    public DetachedTensor(Tensor<T> tensor)
    {
        _data = StopGradient<T>.Detach(tensor);
    }

    /// <inheritdoc />
    public Tensor<T> Data => _data;

    /// <summary>
    /// Implicitly converts a DetachedTensor to its underlying Tensor.
    /// </summary>
    public static implicit operator Tensor<T>(DetachedTensor<T> detached) => detached._data;

    /// <summary>
    /// Creates a DetachedTensor from a regular tensor.
    /// </summary>
    public static DetachedTensor<T> From(Tensor<T> tensor) => new(tensor);
}
