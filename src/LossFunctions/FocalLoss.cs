

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

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clamp predicted values to prevent division by zero using NumericalStabilityHelper
            T p = NumericalStabilityHelper.ClampProbability(predicted[i], NumericalStabilityHelper.SmallEpsilon);

            // pt is the probability of the target class
            T pt = NumOps.Equals(actual[i], NumOps.One) ? p : NumOps.Subtract(NumOps.One, p);

            // alpha term handles class imbalance
            T alphaT = NumOps.Equals(actual[i], NumOps.One) ? _alpha : NumOps.Subtract(NumOps.One, _alpha);

            // (1-pt)^(gamma-1)
            T focusingTerm = NumOps.Power(
                NumOps.Subtract(NumOps.One, pt),
                NumOps.Subtract(_gamma, NumOps.One)
            );

            // Calculate the derivative components
            T term1 = NumOps.Multiply(NumOps.Negate(alphaT), focusingTerm);
            T term2 = NumOps.Subtract(
                NumOps.Multiply(_gamma, NumOps.Subtract(NumOps.One, pt)),
                pt
            );

            // Combine the terms
            derivative[i] = NumOps.Multiply(term1, term2);

            // Apply sign adjustment based on class (positive/negative)
            if (!NumOps.Equals(actual[i], NumOps.One))
            {
                derivative[i] = NumOps.Negate(derivative[i]);
            }
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
