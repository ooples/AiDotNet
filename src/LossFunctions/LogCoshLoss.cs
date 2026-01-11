

using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Log-Cosh loss function, a smooth approximation of Mean Absolute Error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Log-Cosh loss is a smooth approximation of the Mean Absolute Error.
/// It calculates the logarithm of the hyperbolic cosine of the difference between predictions and actual values.
/// 
/// This loss function has several desirable properties:
/// - It's smooth everywhere (unlike Huber loss which has a point where its derivative is not continuous)
/// - It's less affected by outliers than Mean Squared Error
/// - It behaves like Mean Squared Error for small errors and Mean Absolute Error for large errors
/// - Its derivatives are well-defined and bounded, which helps prevent gradient explosions during training
/// 
/// Log-Cosh loss is particularly useful for regression problems where:
/// - You want the smoothness of MSE but with better robustness to outliers
/// - You need stable gradients for model training
/// - You want a compromise between MSE and MAE
/// </para>
/// </remarks>
public class LogCoshLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Log-Cosh loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The Log-Cosh loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        T log2 = NumOps.FromDouble(Math.Log(2));

        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);

            // Numerically stable computation of log(cosh(x))
            // For large |x|, exp(x) or exp(-x) overflows, so we use the identity:
            // log(cosh(x)) = |x| + log(1 + exp(-2*|x|)) - log(2)
            // This is always stable because exp(-2*|x|) <= 1 for any x.
            T absDiff = NumOps.Abs(diff);
            T twoAbsDiff = NumOps.Multiply(NumOps.FromDouble(2), absDiff);
            T expNeg2Abs = NumOps.Exp(NumOps.Negate(twoAbsDiff));
            T logOnePlusExp = NumericalStabilityHelper.SafeLog(
                NumOps.Add(NumOps.One, expNeg2Abs),
                NumericalStabilityHelper.SmallEpsilon
            );

            // log(cosh(x)) = |x| + log(1 + exp(-2*|x|)) - log(2)
            T logCosh = NumOps.Subtract(
                NumOps.Add(absDiff, logOnePlusExp),
                log2
            );

            sum = NumOps.Add(sum, logCosh);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Log-Cosh loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of Log-Cosh loss for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        var result = new T[predicted.Length];
        
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            // Derivative of log(cosh(x)) is tanh(x) = (e^x - e^-x) / (e^x + e^-x)
            T expPos = NumOps.Exp(diff);
            T expNeg = NumOps.Exp(NumOps.Negate(diff));
            result[i] = NumOps.Divide(
                NumOps.Subtract(expPos, expNeg),
                NumOps.Add(expPos, expNeg)
            );
        }
        
        return new Vector<T>(result).Divide(NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates both Log-Cosh loss and gradient on GPU in a single efficient pass.
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
            return base.CalculateLossAndGradientGpu(predicted, actual);
        }

        int size = predicted.ElementCount;

        // Compute loss on GPU
        float lossValue = backend.LogCoshLoss(predicted.Buffer, actual.Buffer, size);

        // Allocate gradient buffer and compute gradient on GPU
        var gradientBuffer = backend.AllocateBuffer(size);
        backend.LogCoshBackward(predicted.Buffer, actual.Buffer, gradientBuffer, size);

        // Create gradient tensor
        var gradientTensor = new GpuTensor<T>(backend, gradientBuffer, predicted.Shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), gradientTensor);
    }
}
