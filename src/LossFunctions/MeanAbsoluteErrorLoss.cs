namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Mean Absolute Error (MAE) loss function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mean Absolute Error measures the average absolute difference between 
/// predicted and actual values.
/// 
/// The formula is: MAE = (1/n) * ?|predicted - actual|
/// 
/// MAE has these key properties:
/// - It treats all errors linearly (unlike MSE which squares errors)
/// - It's less sensitive to outliers than MSE
/// - It's simple to understand as the average magnitude of errors
/// - It's always positive, with perfect predictions giving a value of zero
/// 
/// MAE is ideal for problems where:
/// - You're predicting continuous values
/// - You want all errors to be treated equally (not emphasizing large errors)
/// - The prediction errors follow a Laplace distribution
/// - Outliers should not have a disproportionate influence on the model
/// </para>
/// </remarks>
public class MeanAbsoluteErrorLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Mean Absolute Error between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The mean absolute error value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        return StatisticsHelper<T>.CalculateMeanAbsoluteError(predicted, actual);
    }
    
    /// <summary>
    /// Calculates the derivative of the Mean Absolute Error loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of MAE for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        // The derivative of MAE is sign(predicted-actual)/n
        return predicted.Subtract(actual).Transform(x => 
            NumOps.GreaterThan(x, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One)
        ).Divide(NumOps.FromDouble(predicted.Length));
    }
}