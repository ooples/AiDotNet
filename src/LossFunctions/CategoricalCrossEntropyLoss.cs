namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Categorical Cross Entropy loss function for multi-class classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Categorical Cross Entropy is used for multi-class classification problems,
/// where you need to assign inputs to one of several categories (like classifying images as dog, cat, bird, etc.).
/// 
/// It measures how well the predicted probability distribution matches the actual distribution of classes.
/// 
/// The formula is: CCE = -(1/n) * ?[?(actual_j * log(predicted_j))]
/// 
/// Where:
/// - actual_j is usually a one-hot encoded vector (1 for the correct class, 0 for others)
/// - predicted_j is the predicted probability for each class (typically from a softmax output)
/// - The inner sum is over all classes, and the outer sum is over all samples
/// 
/// Key properties:
/// - Predicted values should be probabilities (between 0 and 1) that sum to 1 across classes
/// - It heavily penalizes confident incorrect predictions
/// - It's the standard loss function for multi-class neural network classifiers
/// - Often used together with the softmax activation function in the output layer
/// 
/// This loss function is ideal when your model needs to choose one option from multiple possibilities.
/// </para>
/// </remarks>
public class CategoricalCrossEntropyLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Small value to prevent numerical instability with log(0).
    /// </summary>
    private readonly T _epsilon;
    
    /// <summary>
    /// Initializes a new instance of the CategoricalCrossEntropyLoss class.
    /// </summary>
    public CategoricalCrossEntropyLoss()
    {
        _epsilon = NumOps.FromDouble(1e-15);
    }
    
    /// <summary>
    /// Calculates the Categorical Cross Entropy loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities that sum to 1 across categories).</param>
    /// <param name="actual">The actual (target) values (typically one-hot encoded).</param>
    /// <returns>The categorical cross entropy loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clamp values to prevent log(0)
            T p = MathHelper.Clamp(predicted[i], _epsilon, NumOps.Subtract(NumOps.One, _epsilon));
            
            // -?(actual * log(predicted))
            sum = NumOps.Add(sum, NumOps.Multiply(actual[i], NumOps.Log(p)));
        }
        
        return NumOps.Negate(sum);
    }
    
    /// <summary>
    /// Calculates the derivative of the Categorical Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities that sum to 1 across categories).</param>
    /// <param name="actual">The actual (target) values (typically one-hot encoded).</param>
    /// <returns>A vector containing the derivatives of CCE for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        // When used with softmax, the derivative simplifies to (predicted - actual)
        return predicted.Subtract(actual).Divide(NumOps.FromDouble(predicted.Length));
    }
}