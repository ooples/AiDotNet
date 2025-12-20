namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Quantile loss function for quantile regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Quantile loss helps predict specific percentiles of data rather than just the average.
/// 
/// For example:
/// - With quantile=0.5, it predicts the median value (50th percentile)
/// - With quantile=0.9, it predicts the 90th percentile
/// - With quantile=0.1, it predicts the 10th percentile
/// 
/// This is useful when you care more about certain parts of the distribution, such as:
/// - Predicting worst-case scenarios (high quantiles)
/// - Ensuring predictions don't fall below a certain threshold (low quantiles)
/// - Creating prediction intervals (by predicting multiple quantiles)
/// 
/// The loss function applies different penalties for overestimation versus underestimation,
/// which forces the model to learn the specific quantile rather than just the average.
/// </para>
/// </remarks>
public class QuantileLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The quantile value to estimate (between 0 and 1).
    /// </summary>
    private readonly T _quantile;

    /// <summary>
    /// Initializes a new instance of the QuantileLoss class.
    /// </summary>
    /// <param name="quantile">The quantile value between 0 and 1 to estimate. Default is 0.5 (median).</param>
    public QuantileLoss(double quantile = 0.5)
    {
        if (quantile < 0 || quantile > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(quantile), "Quantile must be between 0 and 1.");
        }

        _quantile = NumOps.FromDouble(quantile);
    }

    /// <summary>
    /// Calculates the Quantile loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The Quantile loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(actual[i], predicted[i]);
            T loss;

            if (NumOps.GreaterThan(diff, NumOps.Zero))
            {
                // If actual > predicted (underestimation), penalty is quantile * diff
                loss = NumOps.Multiply(_quantile, diff);
            }
            else
            {
                // If actual <= predicted (overestimation), penalty is (1-quantile) * |diff|
                loss = NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, _quantile),
                    NumOps.Negate(diff)
                );
            }

            sum = NumOps.Add(sum, loss);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Quantile loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of Quantile loss for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(actual[i], predicted[i]);

            if (NumOps.GreaterThan(diff, NumOps.Zero))
            {
                // If actual > predicted, derivative is -quantile
                derivative[i] = NumOps.Negate(_quantile);
            }
            else
            {
                // If actual <= predicted, derivative is (1-quantile)
                derivative[i] = NumOps.Subtract(NumOps.One, _quantile);
            }
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
