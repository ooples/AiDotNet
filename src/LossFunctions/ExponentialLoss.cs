namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Exponential Loss function, commonly used in boosting algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Exponential Loss is a loss function that heavily penalizes incorrect predictions,
/// especially those that are far off from the true values.
/// 
/// The formula is: exp(-y * f(x))
/// Where:
/// - y is the true label (usually -1 or 1 for binary classification)
/// - f(x) is the model's prediction
/// 
/// Key properties:
/// - It grows exponentially as the error increases
/// - Correct predictions with high confidence result in values close to zero
/// - Incorrect predictions result in very large values
/// - It's especially sensitive to outliers and misclassifications
/// 
/// Exponential Loss is primarily used in:
/// - AdaBoost and other boosting algorithms
/// - Ensemble methods that need to focus on hard examples
/// - Learning problems where avoiding mistakes is critical
/// 
/// The exponential nature makes the model pay more attention to difficult examples
/// and outliers compared to other loss functions like hinge loss or log loss.
/// </para>
/// </remarks>
public class ExponentialLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Exponential Loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector, typically -1 or 1.</param>
    /// <returns>The exponential loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // exp(-y * f(x))
            T exponent = NumOps.Negate(NumOps.Multiply(actual[i], predicted[i]));
            loss = NumOps.Add(loss, NumOps.Exp(exponent));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Exponential Loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector, typically -1 or 1.</param>
    /// <returns>A vector containing the derivatives of the exponential loss with respect to each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // -y * exp(-y * f(x))
            T exponent = NumOps.Negate(NumOps.Multiply(actual[i], predicted[i]));
            derivative[i] = NumOps.Multiply(
                NumOps.Negate(actual[i]),
                NumOps.Exp(exponent)
            );
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
