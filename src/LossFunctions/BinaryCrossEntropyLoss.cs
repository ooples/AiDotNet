

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Binary Cross Entropy loss function for binary classification problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Binary Cross Entropy is used when classifying data into two categories,
/// such as spam/not-spam, positive/negative sentiment, or disease/no-disease.
/// 
/// The formula is: BCE = -(1/n) * ?[actual * log(predicted) + (1-actual) * log(1-predicted)]
/// 
/// It measures how well predicted probabilities match actual binary outcomes:
/// - When the actual value is 1, it evaluates how close the prediction is to 1
/// - When the actual value is 0, it evaluates how close the prediction is to 0
/// 
/// Key properties:
/// - Predicted values must be probabilities (between 0 and 1)
/// - Actual values are typically 0 or 1 (binary labels)
/// - It heavily penalizes confident mistakes (predicting 0.01 when the true value is 1)
/// - It's the preferred loss function for binary classification problems
/// </para>
/// </remarks>
public class BinaryCrossEntropyLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the BinaryCrossEntropyLoss class.
    /// </summary>
    public BinaryCrossEntropyLoss()
    {
    }

    /// <summary>
    /// Calculates the Binary Cross Entropy loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>The binary cross entropy loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clamp values to prevent log(0) using NumericalStabilityHelper
            T p = NumericalStabilityHelper.ClampProbability(predicted[i], NumericalStabilityHelper.SmallEpsilon);
            T oneMinusP = NumericalStabilityHelper.ClampProbability(NumOps.Subtract(NumOps.One, p), NumericalStabilityHelper.SmallEpsilon);

            // -[y*log(p) + (1-y)*log(1-p)]
            sum = NumOps.Add(sum, NumOps.Add(
                NumOps.Multiply(actual[i], NumericalStabilityHelper.SafeLog(p, NumericalStabilityHelper.SmallEpsilon)),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, actual[i]),
                    NumericalStabilityHelper.SafeLog(oneMinusP, NumericalStabilityHelper.SmallEpsilon)
                )
            ));
        }

        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(predicted.Length)));
    }

    /// <summary>
    /// Calculates the derivative of the Binary Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>A vector containing the derivatives of BCE for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clamp values to prevent division by zero using NumericalStabilityHelper
            T p = NumericalStabilityHelper.ClampProbability(predicted[i], NumericalStabilityHelper.SmallEpsilon);

            // -(y/p - (1-y)/(1-p)) with safe division
            T denominator = NumOps.Multiply(p, NumOps.Subtract(NumOps.One, p));
            derivative[i] = NumericalStabilityHelper.SafeDiv(
                NumOps.Subtract(p, actual[i]),
                denominator,
                NumericalStabilityHelper.SmallEpsilon
            );
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
