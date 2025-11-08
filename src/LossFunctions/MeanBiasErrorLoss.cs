namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Mean Bias Error (MBE) loss function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mean Bias Error is a diagnostic metric that reveals systematic bias in predictions.
/// Unlike MSE or MAE which measure error magnitude, MBE tells you the *direction* of errors.
///
/// The formula is: MBE = (1/n) * Σ(actual - predicted)
///
/// Think of MBE like this:
/// - MBE = 0: Your model is unbiased (errors cancel out)
/// - MBE &gt; 0: Your model tends to under-predict (predictions are too low)
/// - MBE &lt; 0: Your model tends to over-predict (predictions are too high)
///
/// Example scenarios:
/// - Weather forecasting: MBE = +2°C means you're consistently predicting 2 degrees too cold
/// - Price predictions: MBE = -$5,000 means you're consistently overestimating by $5,000
/// - Medical diagnostics: MBE helps detect if a test systematically over/under-estimates values
///
/// Key properties:
/// - Can be positive, negative, or zero (unlike MSE/RMSE which are always non-negative)
/// - Errors of opposite signs cancel each other out
/// - Not sensitive to the magnitude of individual errors
/// - Useful for detecting systematic bias, not for measuring overall accuracy
/// - Often used alongside RMSE or MAE for a complete error analysis
///
/// MBE is ideal for:
/// - Diagnosing systematic prediction bias
/// - Calibrating models that consistently over/under-predict
/// - Quality control in measurement systems
/// - Understanding if your model needs adjustment in a specific direction
///
/// <b>Important:</b> MBE should not be used alone for model evaluation, as positive and negative
/// errors cancel out. A model with large errors in both directions could have MBE ≈ 0.
/// Always use MBE together with metrics like RMSE or MAE.
/// </para>
/// </remarks>
public class MeanBiasErrorLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Mean Bias Error between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The mean bias error value. Positive values indicate under-prediction, negative values indicate over-prediction.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // MBE = mean(actual - predicted)
        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Subtract(actual[i], predicted[i]));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Mean Bias Error loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of MBE for each prediction.</returns>
    /// <remarks>
    /// The derivative is constant: d(MBE)/d(predicted) = -1/n for all elements.
    /// This means each prediction contributes equally to reducing bias, regardless of the current error.
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // The derivative of MBE with respect to predicted is -1/n for all elements
        T derivativeValue = NumOps.Negate(NumOps.Divide(NumOps.One, NumOps.FromDouble(predicted.Length)));

        // Create a vector filled with this constant derivative
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = derivativeValue;
        }

        return derivative;
    }
}
