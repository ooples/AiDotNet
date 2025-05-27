using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;

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
public static class VectorHelper<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

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
    public static Vector<T> CreateVector(int size)
    {
        return new Vector<T>(size);
    }

    /// <summary>
    /// Applies softmax to a vector.
    /// </summary>
    public static Vector<T> Softmax(Vector<T> input)
    {
        var output = new Vector<T>(input.Length);

        // Find max for numerical stability
        T maxValue = input[0];
        for (int i = 1; i < input.Length; i++)
        {
            if (_numOps.GreaterThan(input[i], maxValue))
            {
                maxValue = input[i];
            }
        }

        // Calculate exp(x - max) for each element
        T sumExp = _numOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            T shifted = _numOps.Subtract(input[i], maxValue);
            T expValue = _numOps.Exp(shifted);
            output[i] = expValue;
            sumExp = _numOps.Add(sumExp, expValue);
        }

        // Normalize by sum of exps
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = _numOps.Divide(output[i], sumExp);
        }

        return output;
    }

    /// <summary>
    /// Concatenates multiple vectors into a single vector.
    /// </summary>
    /// <param name="vectors">The list of vectors to concatenate.</param>
    /// <returns>A new vector containing all elements from the input vectors in sequence.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method combines multiple vectors into one long vector.
    /// For example, if you have vectors [1, 2] and [3, 4, 5], concatenating them gives [1, 2, 3, 4, 5].
    /// This is useful when you need to combine parameter vectors from different layers of a neural network.
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when vectors list is null.</exception>
    /// <exception cref="ArgumentException">Thrown when vectors list is empty or contains null vectors.</exception>
    public static Vector<T> Concatenate(IList<Vector<T>> vectors)
    {
        if (vectors == null)
            throw new ArgumentNullException(nameof(vectors));
        
        if (vectors.Count == 0)
            throw new ArgumentException("Cannot concatenate empty list of vectors", nameof(vectors));

        // Calculate total length
        int totalLength = 0;
        for (int i = 0; i < vectors.Count; i++)
        {
            if (vectors[i] == null)
                throw new ArgumentException($"Vector at index {i} is null", nameof(vectors));
            totalLength += vectors[i].Length;
        }

        // Create result vector
        var result = new Vector<T>(totalLength);
        
        // Copy all vectors into result
        int currentIndex = 0;
        for (int i = 0; i < vectors.Count; i++)
        {
            var vector = vectors[i];
            for (int j = 0; j < vector.Length; j++)
            {
                result[currentIndex++] = vector[j];
            }
        }

        return result;
    }

    /// <summary>
    /// Concatenates two vectors into a single vector.
    /// </summary>
    /// <param name="first">The first vector.</param>
    /// <param name="second">The second vector.</param>
    /// <returns>A new vector containing elements from both input vectors.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method combines two vectors into one.
    /// For example, concatenating [1, 2] and [3, 4] gives [1, 2, 3, 4].
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when either vector is null.</exception>
    public static Vector<T> Concatenate(Vector<T> first, Vector<T> second)
    {
        if (first == null)
            throw new ArgumentNullException(nameof(first));
        if (second == null)
            throw new ArgumentNullException(nameof(second));

        var result = new Vector<T>(first.Length + second.Length);
        
        // Copy first vector
        for (int i = 0; i < first.Length; i++)
        {
            result[i] = first[i];
        }
        
        // Copy second vector
        for (int i = 0; i < second.Length; i++)
        {
            result[first.Length + i] = second[i];
        }
        
        return result;
    }
}