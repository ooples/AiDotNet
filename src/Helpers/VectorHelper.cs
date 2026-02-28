using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

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
    public static Vector<T> CreateVector<T>(int size)
    {
        return new Vector<T>(size);
    }

    /// <summary>
    /// Computes the L2 (Euclidean) norm of a vector using hardware-accelerated operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The L2 norm (square root of sum of squares).</returns>
    public static T L2Norm<T>(Vector<T> vector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.FromDouble(Math.Sqrt(
            Math.Max(0, numOps.ToDouble(AiDotNetEngine.Current.DotProduct(vector, vector)))));
    }

    /// <summary>
    /// Returns a new unit-length vector in the same direction as the input.
    /// If the input vector has near-zero norm, a clone of the original is returned.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The input vector to normalize.</param>
    /// <param name="epsilon">Minimum norm threshold below which the vector is not normalized. Default: 1e-10.</param>
    /// <returns>A new normalized vector, or a clone if the norm is below epsilon.</returns>
    public static Vector<T> Normalize<T>(Vector<T> vector, double epsilon = 1e-10)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double norm = Math.Sqrt(Math.Max(0,
            numOps.ToDouble(AiDotNetEngine.Current.DotProduct(vector, vector))));

        if (norm < epsilon)
        {
            var clone = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                clone[i] = vector[i];
            }
            return clone;
        }

        return AiDotNetEngine.Current.Multiply(vector, numOps.FromDouble(1.0 / norm));
    }

    /// <summary>
    /// Normalizes a vector in place, modifying the original vector to have unit length.
    /// If the norm is below epsilon, the vector is left unchanged.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to normalize in place.</param>
    /// <param name="epsilon">Minimum norm threshold. Default: 1e-10.</param>
    public static void NormalizeInPlace<T>(Vector<T> vector, double epsilon = 1e-10)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double norm = Math.Sqrt(Math.Max(0,
            numOps.ToDouble(AiDotNetEngine.Current.DotProduct(vector, vector))));

        if (norm < epsilon) return;

        var normalized = AiDotNetEngine.Current.Multiply(vector, numOps.FromDouble(1.0 / norm));
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = normalized[i];
        }
    }

    /// <summary>
    /// Computes the cosine similarity between two vectors, returning a value in [0, 1].
    /// Uses hardware-accelerated dot product operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <param name="epsilon">Minimum denominator threshold. Default: 1e-10.</param>
    /// <returns>Cosine similarity clamped to [0, 1]. Returns 0 if either vector has near-zero norm.</returns>
    public static double CosineSimilarity<T>(Vector<T> a, Vector<T> b, double epsilon = 1e-10)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;

        double dot = numOps.ToDouble(engine.DotProduct(a, b));
        double normA = numOps.ToDouble(engine.DotProduct(a, a));
        double normB = numOps.ToDouble(engine.DotProduct(b, b));

        double denom = Math.Sqrt(normA * normB);
        return denom > epsilon ? Math.Max(0, dot / denom) : 0;
    }

    /// <summary>
    /// Computes the dot product of two vectors using hardware-accelerated operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The dot product as type T.</returns>
    public static T DotProduct<T>(Vector<T> a, Vector<T> b)
    {
        return AiDotNetEngine.Current.DotProduct(a, b);
    }

    /// <summary>
    /// Computes the Euclidean distance between two vectors using hardware-accelerated operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The Euclidean distance as type T.</returns>
    public static T EuclideanDistance<T>(Vector<T> a, Vector<T> b)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var diff = engine.Subtract(a, b);
        return numOps.FromDouble(Math.Sqrt(
            Math.Max(0, numOps.ToDouble(engine.DotProduct(diff, diff)))));
    }
}
