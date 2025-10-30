namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Cosine Similarity Loss between two vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cosine Similarity measures how similar two vectors are in terms of their orientation,
/// regardless of their magnitude (size).
/// 
/// The formula for cosine similarity is: cos(?) = (A�B)/(||A||�||B||)
/// Where:
/// - A�B is the dot product of vectors A and B
/// - ||A|| and ||B|| are the magnitudes (lengths) of vectors A and B
/// - ? is the angle between vectors A and B
/// 
/// The loss is calculated as 1 - cosine similarity, so:
/// - A value of 0 means the vectors are perfectly aligned (very similar)
/// - A value of 1 means they are perpendicular (no similarity)
/// - A value of 2 means they point in exactly opposite directions
/// 
/// Cosine similarity loss is particularly useful for:
/// - Text similarity tasks (comparing document vectors)
/// - Recommendation systems
/// - Image retrieval
/// - Any task where the direction of vectors matters more than their magnitude
/// 
/// It's often preferred over Euclidean distance when working with high-dimensional sparse vectors.
/// </para>
/// </remarks>
public class CosineSimilarityLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Small value to prevent division by zero.
    /// </summary>
    private readonly T _epsilon;
    
    /// <summary>
    /// Initializes a new instance of the CosineSimilarityLoss class.
    /// </summary>
    public CosineSimilarityLoss()
    {
        _epsilon = NumOps.FromDouble(1e-15);
    }
    
    /// <summary>
    /// Calculates the Cosine Similarity Loss between two vectors.
    /// </summary>
    /// <param name="predicted">The predicted vector from the model.</param>
    /// <param name="actual">The actual (target) vector.</param>
    /// <returns>A scalar value representing the cosine similarity loss.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        T dotProduct = NumOps.Zero;
        T normPredicted = NumOps.Zero;
        T normActual = NumOps.Zero;
        
        for (int i = 0; i < predicted.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(predicted[i], actual[i]));
            normPredicted = NumOps.Add(normPredicted, NumOps.Multiply(predicted[i], predicted[i]));
            normActual = NumOps.Add(normActual, NumOps.Multiply(actual[i], actual[i]));
        }
        
        // Add epsilon to prevent division by zero
        normPredicted = NumOps.Add(normPredicted, _epsilon);
        normActual = NumOps.Add(normActual, _epsilon);
        
        T cosineSimilarity = NumOps.Divide(
            dotProduct,
            NumOps.Multiply(NumOps.Sqrt(normPredicted), NumOps.Sqrt(normActual))
        );
        
        // Loss is 1 - similarity
        return NumOps.Subtract(NumOps.One, cosineSimilarity);
    }
    
    /// <summary>
    /// Calculates the derivative of the Cosine Similarity Loss with respect to the predicted values.
    /// </summary>
    /// <param name="predicted">The predicted vector from the model.</param>
    /// <param name="actual">The actual (target) vector.</param>
    /// <returns>A vector containing the gradient of the loss with respect to each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        
        T dotProduct = NumOps.Zero;
        T normPredicted = NumOps.Zero;
        T normActual = NumOps.Zero;
        
        for (int i = 0; i < predicted.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(predicted[i], actual[i]));
            normPredicted = NumOps.Add(normPredicted, NumOps.Multiply(predicted[i], predicted[i]));
            normActual = NumOps.Add(normActual, NumOps.Multiply(actual[i], actual[i]));
        }
        
        // Add epsilon to prevent division by zero
        normPredicted = NumOps.Add(normPredicted, _epsilon);
        normActual = NumOps.Add(normActual, _epsilon);
        
        T normPredSqrt = NumOps.Sqrt(normPredicted);
        T normProduct = NumOps.Multiply(normPredSqrt, NumOps.Sqrt(normActual));
        
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // ?(cos similarity)/?p_i = (a_i*||p||^2 - p_i*(p�a)) / (||p||^3 * ||a||)
            T numerator = NumOps.Subtract(
                NumOps.Multiply(actual[i], normPredicted),
                NumOps.Multiply(predicted[i], dotProduct)
            );
            
            T denominator = NumOps.Multiply(normProduct, normPredSqrt);
            
            // Derivative of the loss is negative of the derivative of cosine similarity
            derivative[i] = NumOps.Negate(NumOps.Divide(numerator, denominator));
        }
        
        return derivative;
    }
}