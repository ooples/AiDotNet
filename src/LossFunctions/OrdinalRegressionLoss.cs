namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Ordinal Regression Loss function for predicting ordered categories.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Ordinal Regression is used when predicting categories that have a meaningful order.
/// Examples include:
/// - Ratings (poor, fair, good, excellent)
/// - Education levels (elementary, middle, high school, college)
/// - Severity levels (mild, moderate, severe)
/// 
/// Unlike regular classification, ordinal regression takes into account that being off by one category
/// is better than being off by multiple categories. For example, predicting "good" when the actual 
/// rating is "fair" is a smaller error than predicting "excellent".
/// 
/// The ordinal regression loss uses a series of binary classifiers, one for each threshold between
/// adjacent categories. For example, with categories [1,2,3,4,5], there are four classifiers:
/// - Is the rating > 1?
/// - Is the rating > 2?
/// - Is the rating > 3?
/// - Is the rating > 4?
/// 
/// This approach preserves the ordering information in the categories during training.
/// </para>
/// </remarks>
public class OrdinalRegressionLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The number of classes or categories in the ordinal scale.
    /// </summary>
    private readonly int _numClasses;

    /// <summary>
    /// Small value to prevent numerical instability.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// Initializes a new instance of the OrdinalRegressionLoss class.
    /// </summary>
    /// <param name="numClasses">The number of classes or categories in the ordinal scale.</param>
    public OrdinalRegressionLoss(int numClasses)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2 for ordinal regression.");
        }

        _numClasses = numClasses;
        _epsilon = NumOps.FromDouble(1e-15);
    }

    /// <summary>
    /// Calculates the Ordinal Regression Loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector, typically integers representing ordinal categories.</param>
    /// <returns>The ordinal regression loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // For each threshold (numClasses - 1 thresholds)
            for (int j = 0; j < _numClasses - 1; j++)
            {
                // Binary indicator: 1 if actual > j, 0 otherwise
                T indicator = NumOps.GreaterThan(actual[i], NumOps.FromDouble(j)) ?
                    NumOps.One : NumOps.Zero;

                // Binary logistic loss for each threshold: log(1 + exp(-indicator * predicted))
                T expTerm = NumOps.Exp(NumOps.Negate(NumOps.Multiply(indicator, predicted[i])));
                T logTerm = NumOps.Log(NumOps.Add(NumOps.One, expTerm));

                loss = NumOps.Add(loss, logTerm);
            }
        }

        return loss;
    }

    /// <summary>
    /// Calculates the derivative of the Ordinal Regression Loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector, typically integers representing ordinal categories.</param>
    /// <returns>A vector containing the derivatives of the ordinal regression loss with respect to each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < _numClasses - 1; j++)
            {
                // Binary indicator: 1 if actual > j, 0 otherwise
                T indicator = NumOps.GreaterThan(actual[i], NumOps.FromDouble(j)) ?
                    NumOps.One : NumOps.Zero;

                // exp(-indicator * predicted)
                T expTerm = NumOps.Exp(NumOps.Negate(NumOps.Multiply(indicator, predicted[i])));

                // -indicator * exp(-indicator * predicted) / (1 + exp(-indicator * predicted))
                T term = NumOps.Divide(
                    NumOps.Negate(NumOps.Multiply(indicator, expTerm)),
                    NumOps.Add(NumOps.One, expTerm)
                );

                sum = NumOps.Add(sum, term);
            }
            derivative[i] = sum;
        }

        return derivative;
    }
}
