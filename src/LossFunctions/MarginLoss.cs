namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Margin loss function, specifically designed for Capsule Networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Margin loss is a special loss function used in Capsule Networks.
/// 
/// The formula is:
/// T_c * max(0, m+ - ||v_c||)^2 + lambda * (1 - T_c) * max(0, ||v_c|| - m-)^2
/// 
/// Where:
/// - T_c is 1 if class c is present, 0 otherwise
/// - ||v_c|| is the length of the output vector of the capsule for class c
/// - m+ is the upper bound (usually 0.9)
/// - m- is the lower bound (usually 0.1)
/// - lambda is a down-weighting factor (usually 0.5)
/// 
/// Key properties:
/// - Encourages the network to output high values for correct classes
/// - Discourages high outputs for incorrect classes
/// - Helps in learning to represent different aspects of the input
/// 
/// Margin loss is ideal for Capsule Networks because:
/// - It allows multiple classes to be present in the same image
/// - It encourages the network to learn to represent different viewpoints and transformations
/// - It helps in achieving equivariance, a key property of Capsule Networks
/// </para>
/// </remarks>
public class MarginLoss<T> : LossFunctionBase<T>
{
    private readonly T _mPlus;
    private readonly T _mMinus;
    private readonly T _lambda;

    /// <summary>
    /// Initializes a new instance of the MarginLoss class with the specified parameters.
    /// </summary>
    /// <param name="mPlus">The upper bound. Default is 0.9.</param>
    /// <param name="mMinus">The lower bound. Default is 0.1.</param>
    /// <param name="lambda">The down-weighting factor. Default is 0.5.</param>
    public MarginLoss(double mPlus = 0.9, double mMinus = 0.1, double lambda = 0.5)
    {
        _mPlus = NumOps.FromDouble(mPlus);
        _mMinus = NumOps.FromDouble(mMinus);
        _lambda = NumOps.FromDouble(lambda);
    }

    /// <summary>
    /// Calculates the Margin loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The Margin loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T v = predicted[i];
            T y = actual[i];

            T term1 = NumOps.Multiply(y, NumOps.Subtract(_mPlus, v));
            term1 = NumOps.GreaterThan(term1, NumOps.Zero) ? NumOps.Multiply(term1, term1) : NumOps.Zero;

            T term2 = NumOps.Multiply(NumOps.Subtract(NumOps.One, y), NumOps.Subtract(v, _mMinus));
            term2 = NumOps.GreaterThan(term2, NumOps.Zero) ? NumOps.Multiply(term2, term2) : NumOps.Zero;

            loss = NumOps.Add(loss, NumOps.Add(term1, NumOps.Multiply(_lambda, term2)));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Margin loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of Margin loss for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T v = predicted[i];
            T y = actual[i];

            T term1 = NumOps.Multiply(y, NumOps.Subtract(_mPlus, v));
            T term2 = NumOps.Multiply(NumOps.Subtract(NumOps.One, y), NumOps.Subtract(v, _mMinus));

            if (NumOps.GreaterThan(term1, NumOps.Zero))
            {
                derivative[i] = NumOps.Negate(NumOps.Multiply(NumOps.FromDouble(2), term1));
            }
            else if (NumOps.GreaterThan(term2, NumOps.Zero))
            {
                derivative[i] = NumOps.Multiply(_lambda, NumOps.Multiply(NumOps.FromDouble(2), term2));
            }
            else
            {
                derivative[i] = NumOps.Zero;
            }
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}
