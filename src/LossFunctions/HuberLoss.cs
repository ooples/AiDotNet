namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Huber loss function, which combines properties of both MSE and MAE.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Huber loss combines the best properties of Mean Squared Error and Mean Absolute Error.
/// 
/// The formula is:
/// - For errors smaller than delta: 0.5 * error�
/// - For errors larger than delta: delta * (|error| - 0.5 * delta)
/// 
/// Where "error" is the difference between predicted and actual values.
/// 
/// Key properties:
/// - For small errors, it behaves like MSE (quadratic/squared behavior)
/// - For large errors, it behaves like MAE (linear behavior)
/// - Less sensitive to outliers than MSE, but still provides smooth gradients
/// - The delta parameter controls the transition point between quadratic and linear regions
/// 
/// Huber loss is ideal for regression problems where:
/// - You want to balance between MSE and MAE
/// - Your data might contain outliers
/// - You need stable gradients for learning
/// 
/// The delta parameter lets you control the definition of an "outlier" - errors larger than delta
/// are treated as outliers and handled using the more robust linear function.
/// </para>
/// </remarks>
public class HuberLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The threshold parameter that determines the transition between quadratic and linear loss.
    /// </summary>
    private readonly T _delta;
    
    /// <summary>
    /// Initializes a new instance of the HuberLoss class with the specified delta.
    /// </summary>
    /// <param name="delta">The threshold parameter that controls the transition point. Default is 1.0.</param>
    public HuberLoss(double delta = 1.0)
    {
        _delta = NumOps.FromDouble(delta);
    }
    
    /// <summary>
    /// Calculates the Huber loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The Huber loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Abs(NumOps.Subtract(predicted[i], actual[i]));
            
            if (NumOps.LessThanOrEquals(diff, _delta))
            {
                // 0.5 * error�
                sum = NumOps.Add(sum, NumOps.Multiply(
                    NumOps.FromDouble(0.5), 
                    NumOps.Multiply(diff, diff)
                ));
            }
            else
            {
                // delta * (|error| - 0.5 * delta)
                sum = NumOps.Add(sum, NumOps.Subtract(
                    NumOps.Multiply(_delta, diff),
                    NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(_delta, _delta))
                ));
            }
        }
        
        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }
    
    /// <summary>
    /// Calculates the derivative of the Huber loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of Huber loss for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            
            if (NumOps.LessThanOrEquals(NumOps.Abs(diff), _delta))
            {
                // For small errors: diff
                derivative[i] = diff;
            }
            else
            {
                // For large errors: delta * sign(diff)
                derivative[i] = NumOps.Multiply(
                    _delta, 
                    NumOps.GreaterThan(diff, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One)
                );
            }
        }
        
        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}