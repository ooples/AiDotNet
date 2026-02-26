

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Dice loss function, commonly used for image segmentation tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Dice loss measures the overlap between predicted and actual segments in an image.
/// It's based on the Dice coefficient (also known as F1 score), which is a statistical measure of similarity.
/// 
/// The formula is: DiceLoss = 1 - (2 * intersection) / (sum of predicted + sum of actual)
/// 
/// Where:
/// - intersection is the sum of element-wise multiplication of predicted and actual values
/// - A value of 0 means perfect overlap (ideal predictions)
/// - A value of 1 means no overlap at all (worst predictions)
/// 
/// Key properties:
/// - It's ideal for problems where the positive class (what you're trying to detect) is rare
/// - Handles imbalanced data better than cross-entropy in many cases
/// - Focuses on maximizing the overlap between predictions and ground truth
/// - Commonly used in medical image segmentation, satellite imagery, and other segmentation tasks
/// 
/// Unlike cross-entropy, which treats each pixel independently, Dice loss considers the global
/// relationship between predicted and actual masks, which often leads to better segmentation results.
/// </para>
/// </remarks>
public class DiceLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the DiceLoss class.
    /// </summary>
    public DiceLoss()
    {
    }

    /// <summary>
    /// Calculates the Dice loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values (typically probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>The Dice loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T intersection = NumOps.Zero;
        T sumPredicted = NumOps.Zero;
        T sumActual = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, NumOps.Multiply(predicted[i], actual[i]));
            sumPredicted = NumOps.Add(sumPredicted, predicted[i]);
            sumActual = NumOps.Add(sumActual, actual[i]);
        }

        // Use NumericalStabilityHelper.SafeDiv to prevent division by zero
        T denominator = NumOps.Add(sumPredicted, sumActual);
        T diceCoefficient = NumericalStabilityHelper.SafeDiv(
            NumOps.Multiply(NumOps.FromDouble(2), intersection),
            denominator,
            NumericalStabilityHelper.SmallEpsilon
        );

        return NumOps.Subtract(NumOps.One, diceCoefficient);
    }

    /// <summary>
    /// Calculates the derivative of the Dice loss function.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of Dice loss for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T intersection = NumOps.Zero;
        T sumPredicted = NumOps.Zero;
        T sumActual = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, NumOps.Multiply(predicted[i], actual[i]));
            sumPredicted = NumOps.Add(sumPredicted, predicted[i]);
            sumActual = NumOps.Add(sumActual, actual[i]);
        }

        T denominator = NumOps.Add(sumPredicted, sumActual);

        var result = new T[predicted.Length];
        for (int i = 0; i < predicted.Length; i++)
        {
            // Derivative of Dice loss wrt p_i:
            // loss = 1 - 2*intersection/denom
            // d(loss)/dp_i = -2 * (a_i * denom - intersection) / denom^2
            // (quotient rule: d(intersection)/dp_i = a_i, d(denom)/dp_i = 1)
            T numerator = NumOps.Subtract(
                NumOps.Multiply(actual[i], denominator),
                intersection
            );
            result[i] = NumOps.Negate(
                NumericalStabilityHelper.SafeDiv(
                    NumOps.Multiply(NumOps.FromDouble(2.0), numerator),
                    NumOps.Multiply(denominator, denominator),
                    NumericalStabilityHelper.SmallEpsilon
                )
            );
        }

        return new Vector<T>(result);
    }


}
