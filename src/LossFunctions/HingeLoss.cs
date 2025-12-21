namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Hinge loss function commonly used in support vector machines.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Hinge loss is used for binary classification problems, particularly in support vector machines (SVMs).
/// It measures how well your model separates different classes.
/// 
/// The formula is: max(0, 1 - y * f(x)), where:
/// - y is the true label (usually -1 or 1)
/// - f(x) is the model's prediction
/// 
/// Key properties of hinge loss:
/// - It penalizes predictions that are incorrect or not confident enough
/// - It's zero when the prediction is correct and confident (y*f(x) = 1)
/// - It increases linearly when the prediction is incorrect or not confident enough
/// - It encourages the model to find a decision boundary with a large margin between classes
/// 
/// This loss function is ideal for binary classification tasks where you want to maximize
/// the margin between different classes, which often improves generalization to new data.
/// </para>
/// </remarks>
public class HingeLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Hinge loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values, typically -1 or 1.</param>
    /// <returns>The average Hinge loss across all samples.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // Hinge loss: max(0, 1 - y*f(x))
            T margin = NumOps.Subtract(
                NumOps.One,
                NumOps.Multiply(actual[i], predicted[i])
            );

            T loss = MathHelper.Max(NumOps.Zero, margin);
            sum = NumOps.Add(sum, loss);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Hinge loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values, typically -1 or 1.</param>
    /// <returns>A vector containing the derivatives of Hinge loss for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T margin = NumOps.Multiply(actual[i], predicted[i]);

            if (NumOps.LessThan(margin, NumOps.One))
            {
                // If y*f(x) < 1, derivative is -y
                derivative[i] = NumOps.Negate(actual[i]);
            }
            else
            {
                // If y*f(x) >= 1, derivative is 0
                derivative[i] = NumOps.Zero;
            }
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
