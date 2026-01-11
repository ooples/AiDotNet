using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Focal Loss function, which gives more weight to hard-to-classify examples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Focal Loss was designed to handle class imbalance in classification problems,
/// especially for object detection tasks where background examples vastly outnumber foreground objects.
/// 
/// It modifies the standard cross-entropy loss by adding a factor that reduces the loss contribution
/// from easy-to-classify examples and increases the importance of hard-to-classify examples.
/// 
/// The formula is: -a(1-p)^? * log(p) for positive class
///                 -(1-a)p^? * log(1-p) for negative class
/// Where:
/// - p is the model's estimated probability for the correct class
/// - a is a weighting factor that balances positive vs negative examples
/// - ? (gamma) is the focusing parameter that adjusts how much to focus on hard examples
/// 
/// Key properties:
/// - When ?=0, Focal Loss equals Cross-Entropy Loss
/// - Higher ? values increase focus on hard-to-classify examples
/// - a helps handle class imbalance by giving more weight to the minority class
/// 
/// This loss function is ideal for:
/// - Highly imbalanced datasets
/// - One-stage object detectors
/// - Any classification task where easy negatives dominate training
/// </para>
/// </remarks>
public class FocalLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The focusing parameter that down-weights easy examples.
    /// </summary>
    private readonly T _gamma;

    /// <summary>
    /// The weighting factor that balances positive vs negative examples.
    /// </summary>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the FocalLoss class.
    /// </summary>
    /// <param name="gamma">The focusing parameter that down-weights easy examples. Default is 2.0.</param>
    /// <param name="alpha">The weighting factor for positive class. Default is 0.25.</param>
    public FocalLoss(double gamma = 2.0, double alpha = 0.25)
    {
        _gamma = NumOps.FromDouble(gamma);
        _alpha = NumOps.FromDouble(alpha);
    }

    /// <summary>
    /// Calculates the Focal Loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted probabilities from the model.</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>The focal loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clamp predicted values to prevent log(0) using NumericalStabilityHelper
            T p = NumericalStabilityHelper.ClampProbability(predicted[i], NumericalStabilityHelper.SmallEpsilon);

            // pt is the probability of the target class
            T pt = NumOps.Equals(actual[i], NumOps.One) ? p : NumOps.Subtract(NumOps.One, p);

            // alpha term handles class imbalance
            T alphaT = NumOps.Equals(actual[i], NumOps.One) ? _alpha : NumOps.Subtract(NumOps.One, _alpha);

            // (1-pt)^gamma is the focusing term
            T focusingTerm = NumOps.Power(NumOps.Subtract(NumOps.One, pt), _gamma);

            // -a(1-pt)^?log(pt) using SafeLog
            T sampleLoss = NumOps.Multiply(
                NumOps.Negate(alphaT),
                NumOps.Multiply(focusingTerm, NumericalStabilityHelper.SafeLog(pt, NumericalStabilityHelper.SmallEpsilon))
            );

            loss = NumOps.Add(loss, sampleLoss);
        }

        return NumOps.Divide(loss, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Focal Loss with respect to the predicted values.
    /// </summary>
    /// <param name="predicted">The predicted probabilities from the model.</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>A vector containing the gradient of the loss with respect to each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        var result = new T[predicted.Length];
        
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = NumericalStabilityHelper.ClampProbability(predicted[i], NumericalStabilityHelper.SmallEpsilon);
            T y = actual[i];
            
            T pt = NumOps.Equals(y, NumOps.One) ? p : NumOps.Subtract(NumOps.One, p);
            T alphaT = NumOps.Equals(y, NumOps.One) ? _alpha : NumOps.Subtract(NumOps.One, _alpha);
            
            T focusingTerm = NumOps.Power(NumOps.Subtract(NumOps.One, pt), _gamma);
            T logPt = NumericalStabilityHelper.SafeLog(pt, NumericalStabilityHelper.SmallEpsilon);
            
            // Derivative of focal loss with respect to p
            T gammaFactor = NumOps.Multiply(_gamma, NumOps.Power(NumOps.Subtract(NumOps.One, pt), NumOps.Subtract(_gamma, NumOps.One)));
            T term1 = NumOps.Multiply(gammaFactor, logPt);
            T term2 = NumOps.Divide(focusingTerm, pt);
            
            T grad = NumOps.Multiply(alphaT, NumOps.Subtract(term1, term2));
            
            result[i] = NumOps.Equals(y, NumOps.One) ? grad : NumOps.Negate(grad);
        }
        
        return new Vector<T>(result).Divide(NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates both Focal Loss and gradient on GPU in a single efficient pass.
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
        float alpha = Convert.ToSingle(NumOps.ToDouble(_alpha));
        float gamma = Convert.ToSingle(NumOps.ToDouble(_gamma));

        // Compute loss on GPU
        float lossValue = backend.FocalLoss(predicted.Buffer, actual.Buffer, size, alpha, gamma);

        // Allocate gradient buffer and compute gradient on GPU
        var gradientBuffer = backend.AllocateBuffer(size);
        backend.FocalBackward(predicted.Buffer, actual.Buffer, gradientBuffer, size, alpha, gamma);

        // Create gradient tensor
        var gradientTensor = new GpuTensor<T>(backend, gradientBuffer, predicted.Shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), gradientTensor);
    }
}
