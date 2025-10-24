namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Cross Entropy loss function for multi-class classification problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cross-Entropy loss measures how different two probability distributions are.
/// It's commonly used for classification problems where the model outputs probabilities.
/// 
/// The formula is: -?(actual_i * log(predicted_i))
/// 
/// Key properties:
/// - Lower values indicate that the predicted distribution is closer to the actual distribution
/// - It encourages the model to be confident about correct predictions
/// - It heavily penalizes confident but incorrect predictions
/// - It's particularly suited for training classifiers
/// 
/// This loss function is often used in conjunction with the softmax activation function
/// in the output layer for multi-class classification problems.
/// </para>
/// </remarks>
public class CrossEntropyLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Small value to prevent numerical instability with log(0).
    /// </summary>
    private readonly T _epsilon;
    
    /// <summary>
    /// Initializes a new instance of the CrossEntropyLoss class.
    /// </summary>
    public CrossEntropyLoss()
    {
        _epsilon = NumOps.FromDouble(1e-15);
    }
    
    /// <summary>
    /// Calculates the Cross-Entropy loss between predicted and actual probability distributions.
    /// </summary>
    /// <param name="predicted">The predicted probability distribution.</param>
    /// <param name="actual">The actual (target) probability distribution.</param>
    /// <returns>The Cross-Entropy loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clamp predicted values to prevent log(0)
            T p = MathHelper.Clamp(predicted[i], _epsilon, NumOps.Subtract(NumOps.One, _epsilon));
            
            // -?(actual_i * log(predicted_i))
            sum = NumOps.Add(sum, NumOps.Multiply(actual[i], NumOps.Log(p)));
        }
        
        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(predicted.Length)));
    }
    
    /// <summary>
    /// Calculates the derivative of the Cross-Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted probability distribution.</param>
    /// <param name="actual">The actual (target) probability distribution.</param>
    /// <returns>A vector containing the derivative of Cross-Entropy loss for each element.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clamp predicted values to prevent division by zero
            T p = MathHelper.Clamp(predicted[i], _epsilon, NumOps.Subtract(NumOps.One, _epsilon));
            
            // -actual_i / predicted_i
            derivative[i] = NumOps.Divide(NumOps.Negate(actual[i]), p);
        }
        
        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}