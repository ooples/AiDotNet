using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.JitCompiler.IR;

/// <summary>
/// Provides extension methods and utilities for working with tensor shapes in the IR.
/// </summary>
/// <remarks>
/// <para>
/// This class provides helper methods for working with tensor shapes (represented as int[] arrays).
/// It integrates with the existing Tensor&lt;T&gt; infrastructure which already uses int[] for shapes.
/// </para>
/// <para><b>For Beginners:</b> In AiDotNet, tensor shapes are represented as integer arrays.
///
/// For example:
/// - [5] is a vector with 5 elements
/// - [3, 4] is a 3×4 matrix
/// - [2, 3, 4] is a 3D tensor
///
/// This class provides utilities to work with these shapes:
/// - Check if two shapes are compatible for operations
/// - Compute the result shape when broadcasting
/// - Validate shapes
/// - Compare shapes
///
/// These utilities are used by the JIT compiler to understand tensor dimensions
/// and generate optimized code.
/// </para>
/// </remarks>
public static class TensorShapeExtensions
{
    /// <summary>
    /// Computes the total number of elements in a tensor with the given shape.
    /// </summary>
    /// <param name="shape">The tensor shape.</param>
    /// <returns>The total number of elements, or -1 if any dimension is dynamic.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how many total values a tensor holds.
    ///
    /// For example:
    /// - [] has 1 element (scalar - a single number)
    /// - [5] has 5 elements
    /// - [3, 4] has 3 × 4 = 12 elements
    /// - [2, 3, 4] has 2 × 3 × 4 = 24 elements
    ///
    /// If any dimension is -1 (meaning "dynamic" or "unknown"), returns -1.
    /// </para>
    /// </remarks>
    public static int GetElementCount(this int[] shape)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));

        // Scalar (empty shape) has 1 element
        if (shape.Length == 0) return 1;

        int count = 1;
        foreach (var dim in shape)
        {
            if (dim < 0) return -1; // Dynamic dimension
            count *= dim;
        }
        return count;
    }

    /// <summary>
    /// Gets the rank (number of dimensions) of a tensor shape.
    /// </summary>
    /// <param name="shape">The tensor shape.</param>
    /// <returns>The number of dimensions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The rank is how many dimensions the tensor has.
    ///
    /// - [5] has rank 1 (a vector)
    /// - [3, 4] has rank 2 (a matrix)
    /// - [2, 3, 4] has rank 3 (a 3D tensor)
    /// - [] has rank 0 (a scalar - single number)
    /// </para>
    /// </remarks>
    public static int GetRank(this int[] shape) => shape.Length;

    /// <summary>
    /// Checks if this shape is compatible with another shape for broadcasting.
    /// </summary>
    /// <param name="shape1">The first shape.</param>
    /// <param name="shape2">The second shape.</param>
    /// <returns>True if the shapes are compatible for broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows operations between tensors of different shapes by automatically
    /// expanding dimensions. Two shapes are compatible if:
    /// - They have the same rank and all dimensions match, OR
    /// - One dimension is 1 (can be broadcast), OR
    /// - One tensor has fewer dimensions (will be expanded)
    /// </para>
    /// <para><b>For Beginners:</b> Broadcasting lets you do operations on tensors of different sizes.
    ///
    /// For example:
    /// - [3, 4] and [3, 4] are compatible (same shape)
    /// - [3, 4] and [1, 4] are compatible (first dimension broadcasts)
    /// - [3, 4] and [4] are compatible (vector broadcasts across all rows)
    /// - [3, 4] and [3, 5] are NOT compatible (incompatible dimensions)
    ///
    /// This is very useful in neural networks where you often add a bias vector to every
    /// row of a matrix - broadcasting handles this automatically.
    /// </para>
    /// </remarks>
    public static bool IsCompatibleWith(this int[] shape1, int[] shape2)
    {
        if (shape1 == null || shape2 == null) return false;

        // Scalars are compatible with everything
        if (shape1.Length == 0 || shape2.Length == 0) return true;

        // Check from right to left (trailing dimensions)
        int maxRank = Math.Max(shape1.Length, shape2.Length);
        for (int i = 1; i <= maxRank; i++)
        {
            int dim1 = i <= shape1.Length ? shape1[shape1.Length - i] : 1;
            int dim2 = i <= shape2.Length ? shape2[shape2.Length - i] : 1;

            // Dimensions must be equal, one must be 1 (broadcast), or -1 (dynamic)
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1 && dim1 != -1 && dim2 != -1)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Computes the broadcast shape resulting from combining two shapes.
    /// </summary>
    /// <param name="shape1">The first shape.</param>
    /// <param name="shape2">The second shape.</param>
    /// <returns>The broadcast result shape.</returns>
    /// <exception cref="InvalidOperationException">Thrown if shapes are not compatible.</exception>
    /// <remarks>
    /// <para>
    /// The broadcast shape is computed by taking the maximum dimension at each position
    /// when comparing from right to left.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates what shape results when broadcasting two tensors.
    ///
    /// Examples:
    /// - [3, 4] + [3, 4] → [3, 4] (same shape)
    /// - [3, 4] + [1, 4] → [3, 4] (first dimension expands from 1 to 3)
    /// - [3, 4] + [4] → [3, 4] (vector broadcasts to match all rows)
    /// - [5, 3, 4] + [4] → [5, 3, 4] (vector broadcasts across all 5×3 positions)
    ///
    /// The result tells us what shape the output will have after the operation.
    /// </para>
    /// </remarks>
    public static int[] BroadcastWith(this int[] shape1, int[] shape2)
    {
        if (!shape1.IsCompatibleWith(shape2))
        {
            throw new InvalidOperationException(
                $"Shapes [{string.Join(", ", shape1)}] and [{string.Join(", ", shape2)}] " +
                $"are not compatible for broadcasting");
        }

        int maxRank = Math.Max(shape1.Length, shape2.Length);
        int[] resultShape = new int[maxRank];

        for (int i = 1; i <= maxRank; i++)
        {
            int dim1 = i <= shape1.Length ? shape1[shape1.Length - i] : 1;
            int dim2 = i <= shape2.Length ? shape2[shape2.Length - i] : 1;

            // Take maximum (handle dynamic dimensions)
            if (dim1 == -1 || dim2 == -1)
            {
                resultShape[maxRank - i] = -1; // Dynamic
            }
            else
            {
                resultShape[maxRank - i] = Math.Max(dim1, dim2);
            }
        }

        return resultShape;
    }

    /// <summary>
    /// Checks if two shapes are exactly equal.
    /// </summary>
    /// <param name="shape1">The first shape.</param>
    /// <param name="shape2">The second shape.</param>
    /// <returns>True if shapes are equal.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks if two shapes are identical.
    ///
    /// Examples:
    /// - [3, 4] equals [3, 4] → true
    /// - [3, 4] equals [4, 3] → false (different order!)
    /// - [3, 4] equals [1, 4] → false (different dimensions)
    /// </para>
    /// </remarks>
    public static bool ShapesEqual(int[]? shape1, int[]? shape2)
    {
        if (ReferenceEquals(shape1, shape2)) return true;
        if (shape1 == null || shape2 == null) return false;
        if (shape1.Length != shape2.Length) return false;

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Creates a string representation of a shape.
    /// </summary>
    /// <param name="shape">The shape to represent.</param>
    /// <returns>A string representation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a shape to a readable string for debugging.
    ///
    /// Examples:
    /// - [] → "scalar"
    /// - [5] → "[5]"
    /// - [3, 4] → "[3, 4]"
    /// - [2, -1, 4] → "[2, ?, 4]" (? means dynamic)
    /// </para>
    /// </remarks>
    public static string ShapeToString(this int[] shape)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));

        if (shape.Length == 0) return "scalar";
        return $"[{string.Join(", ", shape.Select(d => d >= 0 ? d.ToString() : "?"))}]";
    }

    /// <summary>
    /// Computes a hash code for a tensor shape.
    /// </summary>
    /// <param name="shape">The shape to hash.</param>
    /// <returns>A hash code.</returns>
    /// <remarks>
    /// <para>
    /// This hash code can be used to cache compiled graphs based on shape.
    /// Shapes with the same dimensions will have the same hash.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a unique number that represents the shape.
    ///
    /// It's like a fingerprint for the shape - two identical shapes will have
    /// the same hash code. This is used to quickly check if we've already compiled
    /// code for a tensor of this shape, so we can reuse it instead of recompiling.
    /// </para>
    /// </remarks>
    public static int GetShapeHashCode(this int[] shape)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));

        int hash = 17;
        foreach (var dim in shape)
        {
            hash = hash * 31 + dim.GetHashCode();
        }
        return hash;
    }

    /// <summary>
    /// Extracts the shape from a Tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The tensor.</param>
    /// <returns>The shape as an int array.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gets the shape from an existing Tensor object.
    ///
    /// Since Tensor already has a Shape property, this just returns it.
    /// It's provided for consistency with the IR infrastructure.
    /// </para>
    /// </remarks>
    public static int[] GetShape<T>(this Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Return a defensive copy to prevent mutation of internal state
        return tensor.Shape.ToArray();
    }

    /// <summary>
    /// Validates that a shape is well-formed.
    /// </summary>
    /// <param name="shape">The shape to validate.</param>
    /// <returns>True if valid.</returns>
    /// <remarks>
    /// <para>
    /// A shape is valid if all dimensions are either positive or -1 (dynamic).
    /// Zero dimensions are not allowed.
    /// </para>
    /// <para><b>For Beginners:</b> This checks that a shape makes sense.
    ///
    /// Valid shapes:
    /// - [] (scalar)
    /// - [5] (vector with 5 elements)
    /// - [3, 4] (3×4 matrix)
    /// - [-1, 4] (dynamic first dimension, 4 columns)
    ///
    /// Invalid shapes:
    /// - [0, 4] (can't have zero dimension)
    /// - [3, -2] (only -1 is allowed for dynamic)
    /// </para>
    /// </remarks>
    public static bool IsValidShape(this int[] shape)
    {
        if (shape == null) return false;

        foreach (var dim in shape.Where(d => d <= 0 && d != -1))
        {
            // Dimensions must be positive or -1 (dynamic)
            return false;
        }

        return true;
    }
}
