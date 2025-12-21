

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Poisson loss function for count data modeling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Poisson loss is designed for modeling count data where the target values
/// represent the number of occurrences of an event in a fixed interval.
/// 
/// Examples include:
/// - Number of customer arrivals per hour
/// - Number of network failures per day
/// - Number of disease cases per region
/// 
/// The loss is derived from the Poisson probability distribution, which is ideal for modeling
/// rare events where we know the average rate of occurrence.
/// 
/// The formula is: predicted - actual * log(predicted) + log(actual!)
/// 
/// Since log(actual!) is constant with respect to predictions, it can be omitted during optimization,
/// so the loss is often implemented as just: predicted - actual * log(predicted)
/// 
/// Poisson loss is appropriate when:
/// - Your target values are non-negative counts
/// - The variance of the data is approximately equal to the mean
/// - You're modeling the rate or frequency of events
/// </para>
/// </remarks>
public class PoissonLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the PoissonLoss class.
    /// </summary>
    public PoissonLoss()
    {
    }

    /// <summary>
    /// Calculates the Poisson loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model (should be positive).</param>
    /// <param name="actual">The actual (target) values (typically counts or rates).</param>
    /// <returns>The average Poisson loss across all samples.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // Poisson loss: predicted - actual * log(predicted)
            // (Omitting log(actual!) as it's constant wrt predictions)
            sum = NumOps.Add(sum, NumOps.Subtract(
                predicted[i],
                NumOps.Multiply(actual[i], NumericalStabilityHelper.SafeLog(predicted[i], NumericalStabilityHelper.SmallEpsilon))
            ));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Poisson loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model (should be positive).</param>
    /// <param name="actual">The actual (target) values (typically counts or rates).</param>
    /// <returns>A vector containing the gradient of the loss with respect to each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // The derivative is 1 - actual/predicted
            derivative[i] = NumOps.Subtract(NumOps.One, NumericalStabilityHelper.SafeDiv(actual[i], predicted[i], NumericalStabilityHelper.SmallEpsilon));
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
