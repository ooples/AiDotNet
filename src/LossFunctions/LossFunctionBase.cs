using AiDotNet.Tensors.Engines.Gpu;

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
    /// Calculates both loss and gradient on GPU in a single pass.
    /// Default implementation falls back to separate calls.
    /// </summary>
    /// <param name="predicted">The predicted GPU tensor from the model.</param>
    /// <param name="actual">The actual (target) GPU tensor.</param>
    /// <returns>A tuple containing the loss value and gradient tensor.</returns>
    public virtual (T Loss, IGpuTensor<T> Gradient) CalculateLossAndGradientGpu(IGpuTensor<T> predicted, IGpuTensor<T> actual)
    {
        // Default: fall back to CPU
        var predictedCpu = predicted.ToTensor();
        var actualCpu = actual.ToTensor();
        
        var loss = CalculateLoss(predictedCpu.ToVector(), actualCpu.ToVector());
        var gradientCpu = CalculateDerivative(predictedCpu.ToVector(), actualCpu.ToVector());
        
        var gradientTensor = new Tensor<T>(predictedCpu.Shape);
        Array.Copy(gradientCpu.ToArray(), gradientTensor.Data, gradientCpu.Length);
        
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");
        var gradientGpu = new GpuTensor<T>(backend, gradientTensor, GpuTensorRole.Gradient);
        
        return (loss, gradientGpu);
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
