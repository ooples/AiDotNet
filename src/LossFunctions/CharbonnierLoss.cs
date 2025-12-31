namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Charbonnier loss function, a smooth approximation of L1 loss.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Charbonnier loss is a differentiable approximation of the absolute error (L1 loss).
///
/// The formula is: L(x, y) = sqrt((x - y)² + ε²)
///
/// Where:
/// - x is the predicted value
/// - y is the actual value
/// - ε (epsilon) is a small constant (typically 1e-6 or 1e-9)
///
/// Key properties:
/// - Like L1 loss, it's robust to outliers
/// - Unlike L1 loss, it's differentiable everywhere (smooth at zero)
/// - Provides more stable gradients for training deep neural networks
/// - Widely used in video super-resolution and image restoration
///
/// Charbonnier loss is preferred over L1 loss in deep learning because:
/// - L1 loss has an undefined derivative at zero
/// - Charbonnier loss provides smooth gradients that help with optimization
/// - The epsilon parameter controls the "sharpness" of the approximation
/// </para>
/// <para>
/// <b>Reference:</b> Charbonnier et al., "Two deterministic half-quadratic regularization algorithms
/// for computed imaging", ICIP 1994.
/// </para>
/// </remarks>
public class CharbonnierLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Small constant to ensure differentiability at zero.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// The squared epsilon value, precomputed for efficiency.
    /// </summary>
    private readonly T _epsilonSquared;

    /// <summary>
    /// Initializes a new instance of the CharbonnierLoss class.
    /// </summary>
    /// <param name="epsilon">The smoothing parameter. Default is 1e-6.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The epsilon parameter controls how closely Charbonnier loss
    /// approximates L1 loss:
    /// - Smaller epsilon: Closer to L1 loss but less smooth
    /// - Larger epsilon: Smoother gradients but less like L1 loss
    ///
    /// The default value of 1e-6 works well for most applications.
    /// </para>
    /// </remarks>
    public CharbonnierLoss(double epsilon = 1e-6)
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _epsilonSquared = NumOps.FromDouble(epsilon * epsilon);
    }

    /// <summary>
    /// Calculates the Charbonnier loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The Charbonnier loss value.</returns>
    /// <remarks>
    /// <para>
    /// The loss is computed as the mean of sqrt((predicted - actual)² + ε²)
    /// across all elements.
    /// </para>
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            T diffSquared = NumOps.Multiply(diff, diff);
            // sqrt(diff² + ε²)
            T charbonnier = NumOps.Sqrt(NumOps.Add(diffSquared, _epsilonSquared));
            sum = NumOps.Add(sum, charbonnier);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Charbonnier loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of Charbonnier loss for each prediction.</returns>
    /// <remarks>
    /// <para>
    /// The derivative is: (predicted - actual) / sqrt((predicted - actual)² + ε²)
    ///
    /// This derivative is always well-defined (unlike L1 loss) because the denominator
    /// is always at least ε, avoiding division by zero.
    /// </para>
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        T n = NumOps.FromDouble(predicted.Length);

        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            T diffSquared = NumOps.Multiply(diff, diff);
            // diff / sqrt(diff² + ε²)
            T denominator = NumOps.Sqrt(NumOps.Add(diffSquared, _epsilonSquared));
            derivative[i] = NumOps.Divide(NumOps.Divide(diff, denominator), n);
        }

        return derivative;
    }
}
