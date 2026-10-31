using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// The "loss" Σ predicted ⊙ target, whose parameter gradient is the vector-Jacobian product Jᵀ·target.
/// </summary>
/// <remarks>
/// Handing this to a network's <c>ComputeGradients</c> with a weight tensor as the target returns
/// Jᵀw in the network's <c>GetParameters()</c> layout, through the network's own tape. Agents that need a
/// gradient whose output-space form they know in closed form (TRPO's surrogate gradient and Fisher-vector
/// products) use it instead of re-deriving the backward pass by hand.
/// </remarks>
internal sealed class OutputContractionLoss<T> : ILossFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>The shared stateless instance.</summary>
    public static OutputContractionLoss<T> Instance { get; } = new OutputContractionLoss<T>();

    /// <inheritdoc/>
    public T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actual is null) throw new ArgumentNullException(nameof(actual));
        if (predicted.Length != actual.Length)
        {
            throw new ArgumentException("Predicted and weight vectors must have the same length.", nameof(actual));
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(predicted[i], actual[i]));
        }
        return sum;
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var product = Engine.TensorMultiply(predicted, target);
        var allAxes = Enumerable.Range(0, product.Shape.Length).ToArray();
        return Engine.ReduceSum(product, allAxes, keepDims: false);
    }

    /// <inheritdoc/>
    public (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actual is null) throw new ArgumentNullException(nameof(actual));

        // d(sum predicted * actual) / d predicted = actual.
        return (CalculateLoss(predicted.ToVector(), actual.ToVector()), actual);
    }
}
