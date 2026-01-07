namespace AiDotNet.LossFunctions;

/// <summary>
/// Base class for loss function implementations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public abstract class LossFunctionBase<T> : ILossFunction<T>
{
    /// <summary>
    /// Provides operations for the numeric type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Initializes a new instance of the LossFunctionBase class.
    /// </summary>
    protected LossFunctionBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The loss value.</returns>
    public abstract T CalculateLoss(Vector<T> predicted, Vector<T> actual);

    /// <summary>
    /// Calculates the derivative (gradient) of the loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of the loss with respect to each prediction.</returns>
    public abstract Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual);

    /// <summary>
    /// Calculates the loss between predicted and actual tensors on GPU.
    /// Default implementation falls back to CPU.
    /// </summary>
    /// <param name="predicted">The predicted tensor from the model (on GPU).</param>
    /// <param name="actual">The actual (target) tensor (on GPU).</param>
    /// <returns>The loss value.</returns>
    public virtual T CalculateLossGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        // Default: fall back to CPU
        return CalculateLoss(predicted.ToVector(), actual.ToVector());
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the loss function on GPU.
    /// Default implementation falls back to CPU.
    /// </summary>
    /// <param name="predicted">The predicted tensor from the model (on GPU).</param>
    /// <param name="actual">The actual (target) tensor (on GPU).</param>
    /// <returns>A tensor containing the derivatives of the loss with respect to each prediction (on GPU).</returns>
    public virtual Tensor<T> CalculateDerivativeGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        // Default: fall back to CPU
        var derivative = CalculateDerivative(predicted.ToVector(), actual.ToVector());
        var result = new Tensor<T>(predicted.Shape);
        Array.Copy(derivative.ToArray(), result.Data, derivative.Length);
        return result;
    }

    /// <summary>
    /// Validates that the predicted and actual vectors have the same length.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual values vector.</param>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    protected void ValidateVectorLengths(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
        {
            throw new ArgumentException("Predicted and actual vectors must have the same length.");
        }
    }
}
