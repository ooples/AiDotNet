namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Modified Huber Loss function, a smoother version of the hinge loss.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Modified Huber Loss is a smoother version of the hinge loss that's less sensitive to outliers.
/// It combines quadratic behavior near zero with linear behavior for large negative values.
/// 
/// The formula is:
/// - For z = -1: max(0, 1 - z)²
/// - For z &lt; -1: -4 * z
/// 
/// Where z = y * f(x), with y being the true label and f(x) the prediction.
/// 
/// Key properties:
/// - It's smoother than hinge loss, making optimization easier
/// - It's more robust to outliers than squared hinge loss
/// - It combines the benefits of both quadratic and linear losses
/// - It has a continuous first derivative
/// 
/// Modified Huber Loss is particularly useful for:
/// - Binary classification problems
/// - Datasets with noisy labels
/// - Problems where you want to balance between being sensitive to errors but not overly influenced by extreme mistakes
/// </para>
/// </remarks>
public class ModifiedHuberLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Modified Huber Loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector, typically -1 or 1.</param>
    /// <returns>The modified huber loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // z = y * f(x)
            T z = NumOps.Multiply(actual[i], predicted[i]);

            if (NumOps.GreaterThanOrEquals(z, NumOps.FromDouble(-1)))
            {
                // For z = -1: max(0, 1 - z)²
                T margin = NumOps.Subtract(NumOps.One, z);
                T hingeLoss = MathHelper.Max(NumOps.Zero, margin);
                loss = NumOps.Add(loss, NumOps.Power(hingeLoss, NumOps.FromDouble(2)));
            }
            else
            {
                // For z < -1: -4 * z
                loss = NumOps.Add(loss, NumOps.Multiply(NumOps.FromDouble(-4), z));
            }
        }

        return NumOps.Divide(loss, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Modified Huber Loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector, typically -1 or 1.</param>
    /// <returns>A vector containing the derivatives of the modified huber loss with respect to each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // z = y * f(x)
            T z = NumOps.Multiply(actual[i], predicted[i]);

            if (NumOps.GreaterThanOrEquals(z, NumOps.FromDouble(-1)))
            {
                if (NumOps.LessThan(z, NumOps.One))
                {
                    // For -1 = z < 1: -2 * y * (1 - z)
                    derivative[i] = NumOps.Multiply(
                        NumOps.Multiply(NumOps.FromDouble(-2), actual[i]),
                        NumOps.Subtract(NumOps.One, z)
                    );
                }
                else
                {
                    // For z = 1: 0
                    derivative[i] = NumOps.Zero;
                }
            }
            else
            {
                // For z < -1: -4 * y
                derivative[i] = NumOps.Multiply(NumOps.FromDouble(-4), actual[i]);
            }
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
