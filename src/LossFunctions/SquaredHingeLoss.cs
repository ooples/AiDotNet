using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Squared Hinge Loss function for binary classification problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Squared Hinge Loss is a variation of the Hinge Loss used in Support Vector Machines (SVMs)
/// that applies a squared penalty to incorrectly classified examples.
/// 
/// The formula is: max(0, 1 - y*f(x))²
/// Where:
/// - y is the true label (usually -1 or 1)
/// - f(x) is the model's prediction
/// 
/// Key properties:
/// - It heavily penalizes predictions that are incorrect or not confident enough
/// - The quadratic nature creates a smoother loss surface compared to regular Hinge Loss
/// - It has a continuous derivative everywhere, which can make optimization easier
/// - It's zero when predictions are correct and confident (y*f(x) = 1)
/// 
/// Squared Hinge Loss is particularly useful for:
/// - Binary classification problems
/// - Support Vector Machines
/// - Any situation where smoother gradients are beneficial for optimization
/// 
/// Compared to regular Hinge Loss, it penalizes violations more severely due to the squaring operation.
/// </para>
/// </remarks>
public class SquaredHingeLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Squared Hinge Loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector, typically -1 or 1.</param>
    /// <returns>The squared hinge loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // Calculate margin: 1 - y*f(x)
            T margin = NumOps.Subtract(
                NumOps.One,
                NumOps.Multiply(actual[i], predicted[i])
            );

            // Apply squared hinge: max(0, margin)²
            T hingeLoss = MathHelper.Max(NumOps.Zero, margin);
            loss = NumOps.Add(loss, NumOps.Power(hingeLoss, NumOps.FromDouble(2)));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Squared Hinge Loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector, typically -1 or 1.</param>
    /// <returns>A vector containing the derivatives of the squared hinge loss with respect to each predicted value.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        var result = new T[predicted.Length];
        
        for (int i = 0; i < predicted.Length; i++)
        {
            T margin = NumOps.Subtract(NumOps.One, NumOps.Multiply(actual[i], predicted[i]));
            
            if (NumOps.GreaterThan(margin, NumOps.Zero))
            {
                // Derivative is -2 * margin * y when margin > 0
                result[i] = NumOps.Multiply(
                    NumOps.FromDouble(-2.0),
                    NumOps.Multiply(margin, actual[i])
                );
            }
            else
            {
                // Derivative is 0 when margin <= 0
                result[i] = NumOps.Zero;
            }
        }
        
        return new Vector<T>(result).Divide(NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates both Squared Hinge loss and gradient on GPU in a single efficient pass.
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
        float lossValue = backend.SquaredHingeLoss(predicted.Buffer, actual.Buffer, size);

        // Allocate gradient buffer and compute gradient on GPU
        var gradientBuffer = backend.AllocateBuffer(size);
        backend.SquaredHingeBackward(predicted.Buffer, actual.Buffer, gradientBuffer, size);

        // Create gradient tensor
        var gradientTensor = new GpuTensor<T>(backend, gradientBuffer, predicted.Shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), gradientTensor);
    }
}
