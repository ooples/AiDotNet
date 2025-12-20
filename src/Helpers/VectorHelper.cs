namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for creating and manipulating vectors used in AI and machine learning operations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In AI and machine learning, a vector is simply a list of numbers arranged in a specific order.
/// Think of it as a one-dimensional array or a single column/row of data. Vectors are used to represent:
/// - Features of a single data point (like height, weight, age of a person)
/// - Target values we want to predict
/// - Weights in a trained model
/// - Intermediate calculations during model training
/// 
/// This helper class provides convenient methods to work with vectors in your AI applications.
/// </remarks>
public static class VectorHelper
{
    /// <summary>
    /// Creates a new vector with the specified size.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements (e.g., double, float).</typeparam>
    /// <param name="size">The number of elements in the vector.</param>
    /// <returns>A new vector initialized with default values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method creates an empty vector with a specific length.
    /// For example, if you need a vector to store 5 values, you would call:
    /// <code>
    /// var myVector = VectorHelper.CreateVector&lt;double&gt;(5);
    /// </code>
    /// This creates a vector that can hold 5 double values, initially set to 0.
    /// You can then assign values to individual elements using indexing:
    /// <code>
    /// myVector[0] = 10.5;
    /// myVector[1] = 20.3;
    /// </code>
    /// </remarks>
    public static Vector<T> CreateVector<T>(int size)
    {
        return new Vector<T>(size);
    }
}
