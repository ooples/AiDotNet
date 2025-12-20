

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
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            // log(cosh(x)) = log((e^x + e^-x)/2)
            T logCosh = NumericalStabilityHelper.SafeLog(
                NumOps.Divide(
                    NumOps.Add(
                        NumOps.Exp(diff),
                        NumOps.Exp(NumOps.Negate(diff))
                    ),
                    NumOps.FromDouble(2)
                ),
                NumericalStabilityHelper.SmallEpsilon
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

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            // The derivative of log(cosh(x)) is tanh(x)
            derivative[i] = MathHelper.Tanh(diff);
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
