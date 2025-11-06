namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Root Mean Squared Error (RMSE) loss function.
/// </summary>
/// <typeparam name="T">The numeric type (float or double).</typeparam>
/// <remarks>
/// RMSE measures the square root of the average squared differences between predicted and actual values.
/// It is particularly useful for regression problems and gives more weight to larger errors.
///
/// Formula: RMSE = sqrt(mean((predicted - actual)^2))
///
/// The derivative with respect to predicted values is:
/// d(RMSE)/d(predicted) = (predicted - actual) / (n * RMSE)
/// where n is the number of samples and RMSE is the loss value.
///
/// This implementation leverages the existing StatisticsHelper.CalculateRootMeanSquaredError() method
/// for efficient and consistent calculation across the library.
/// </remarks>
public class RootMeanSquaredErrorLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Root Mean Squared Error loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (ground truth) values.</param>
    /// <returns>The RMSE loss value.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        return StatisticsHelper<T>.CalculateRootMeanSquaredError(actual, predicted);
    }

    /// <summary>
    /// Calculates the derivative of the RMSE loss with respect to predicted values.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (ground truth) values.</param>
    /// <returns>A vector of gradients for each predicted value.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Calculate RMSE for use in derivative calculation
        T rmse = StatisticsHelper<T>.CalculateRootMeanSquaredError(actual, predicted);

        // Avoid division by zero - if RMSE is zero, all predictions are perfect
        if (NumOps.Equals(rmse, NumOps.Zero))
        {
            return Vector<T>.Build.Dense(predicted.Length, NumOps.Zero);
        }

        // The derivative of RMSE is: (predicted - actual) / (n * RMSE)
        T denominator = NumOps.Multiply(NumOps.FromDouble(predicted.Length), rmse);

        return predicted.Subtract(actual).Divide(denominator);
    }
}
