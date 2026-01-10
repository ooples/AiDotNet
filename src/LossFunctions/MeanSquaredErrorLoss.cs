using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Mean Squared Error (MSE) loss function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mean Squared Error is one of the most common loss functions used in regression problems.
/// It measures the average squared difference between predicted and actual values.
/// 
/// The formula is: MSE = (1/n) * ?(predicted - actual)²
/// 
/// MSE has these key properties:
/// - It heavily penalizes large errors due to the squaring operation
/// - It treats all data points equally
/// - It's differentiable everywhere, making it suitable for gradient-based optimization
/// - It's always positive, with perfect predictions giving a value of zero
/// 
/// MSE is ideal for problems where:
/// - You're predicting continuous values (like prices, temperatures, etc.)
/// - Outliers should be given extra attention (due to the squaring)
/// - The prediction errors follow a normal distribution
/// </para>
/// </remarks>
public class MeanSquaredErrorLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Mean Squared Error between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The mean squared error value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        return StatisticsHelper<T>.CalculateMeanSquaredError(predicted, actual);
    }

    /// <summary>
    /// Calculates the derivative of the Mean Squared Error loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of MSE for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // The derivative of MSE is 2*(predicted-actual)/n
        return predicted.Subtract(actual).Transform(x =>
            NumOps.Multiply(NumOps.FromDouble(2), x)
        ).Divide(NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates both MSE loss and gradient on GPU in a single efficient pass.
    /// </summary>
    /// <param name="predicted">The predicted GPU tensor from the model.</param>
    /// <param name="actual">The actual (target) GPU tensor.</param>
    /// <returns>A tuple containing the loss value and gradient tensor.</returns>
    public override (T Loss, IGpuTensor<T> Gradient) CalculateLossAndGradientGpu(IGpuTensor<T> predicted, IGpuTensor<T> actual)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend();

        if (backend == null)
        {
            // Fall back to CPU if GPU backend not available
            return base.CalculateLossAndGradientGpu(predicted, actual);
        }

        int size = predicted.ElementCount;

        // Compute loss on GPU
        float lossValue = backend.MseLoss(predicted.Buffer, actual.Buffer, size);

        // Allocate gradient buffer and compute gradient on GPU
        var gradientBuffer = backend.AllocateBuffer(size);
        backend.MseBackward(predicted.Buffer, actual.Buffer, gradientBuffer, size);

        // Create gradient tensor
        var gradientTensor = new GpuTensor<T>(backend, gradientBuffer, predicted.Shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), gradientTensor);
    }
}
