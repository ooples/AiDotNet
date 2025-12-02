using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Helper methods for vectorized operations in SIMD-optimized code generation.
/// </summary>
/// <remarks>
/// <para>
/// This class provides utility methods for working with Vector, Matrix, and Tensor objects
/// in a SIMD-friendly way. It uses the INumericOperations interface for type-safe arithmetic
/// operations and leverages TensorPrimitives for hardware-accelerated computations when available.
/// </para>
/// <para><b>For Beginners:</b> When we use SIMD (processing multiple numbers at once),
/// we need helper functions for common operations like:
///
/// - Loading chunks of data into SIMD registers
/// - Reducing multiple values to a single result (sum, max, min)
/// - Applying activation functions to vectors and tensors
///
/// These helpers make it easy to write SIMD-optimized code without dealing
/// with low-level vector operations directly.
/// </para>
/// </remarks>
public static class VectorHelper
{
    #region Reduction Operations (Vector-based)

    /// <summary>
    /// Performs horizontal sum reduction on a Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector of values to sum.</param>
    /// <returns>The sum of all elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds up all the numbers in a vector.
    /// For example, HorizontalReduceSum([1, 2, 3, 4]) = 10.
    /// </para>
    /// </remarks>
    public static T HorizontalReduceSum<T>(Vector<T> vector)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Sum(vector.AsSpan());
    }

    /// <summary>
    /// Performs horizontal max reduction on a Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector of values.</param>
    /// <returns>The maximum value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This finds the largest number in a vector.
    /// For example, HorizontalReduceMax([3, 1, 4, 1, 5]) = 5.
    /// </para>
    /// </remarks>
    public static T HorizontalReduceMax<T>(Vector<T> vector)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Max(vector.AsSpan());
    }

    /// <summary>
    /// Performs horizontal min reduction on a Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector of values.</param>
    /// <returns>The minimum value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This finds the smallest number in a vector.
    /// For example, HorizontalReduceMin([3, 1, 4, 1, 5]) = 1.
    /// </para>
    /// </remarks>
    public static T HorizontalReduceMin<T>(Vector<T> vector)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Min(vector.AsSpan());
    }

    /// <summary>
    /// Performs horizontal mean reduction on a Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector of values.</param>
    /// <returns>The mean (average) value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates the average of all numbers in a vector.
    /// For example, HorizontalReduceMean([2, 4, 6, 8]) = 5.
    /// </para>
    /// </remarks>
    public static T HorizontalReduceMean<T>(Vector<T> vector)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var sum = ops.Sum(vector.AsSpan());
        return ops.Divide(sum, ops.FromDouble(vector.Length));
    }

    #endregion

    #region Reduction Operations (Tensor-based)

    /// <summary>
    /// Performs horizontal sum reduction on a Tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor of values to sum.</param>
    /// <returns>The sum of all elements.</returns>
    public static T HorizontalReduceSum<T>(Tensor<T> tensor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Sum(tensor.AsSpan());
    }

    /// <summary>
    /// Performs horizontal max reduction on a Tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor of values.</param>
    /// <returns>The maximum value.</returns>
    public static T HorizontalReduceMax<T>(Tensor<T> tensor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Max(tensor.AsSpan());
    }

    /// <summary>
    /// Performs horizontal min reduction on a Tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor of values.</param>
    /// <returns>The minimum value.</returns>
    public static T HorizontalReduceMin<T>(Tensor<T> tensor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Min(tensor.AsSpan());
    }

    /// <summary>
    /// Performs horizontal mean reduction on a Tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor of values.</param>
    /// <returns>The mean (average) value.</returns>
    public static T HorizontalReduceMean<T>(Tensor<T> tensor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var sum = ops.Sum(tensor.AsSpan());
        return ops.Divide(sum, ops.FromDouble(tensor.Length));
    }

    #endregion

    #region Reduction Operations (Array-based for Expression Trees)

    /// <summary>
    /// Performs horizontal sum reduction on an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="array">The array of values to sum.</param>
    /// <returns>The sum of all elements.</returns>
    public static T HorizontalReduceSumArray<T>(T[] array)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Sum(array.AsSpan());
    }

    /// <summary>
    /// Performs horizontal max reduction on an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="array">The array of values.</param>
    /// <returns>The maximum value.</returns>
    public static T HorizontalReduceMaxArray<T>(T[] array)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Max(array.AsSpan());
    }

    /// <summary>
    /// Performs horizontal min reduction on an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="array">The array of values.</param>
    /// <returns>The minimum value.</returns>
    public static T HorizontalReduceMinArray<T>(T[] array)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Min(array.AsSpan());
    }

    /// <summary>
    /// Performs horizontal mean reduction on an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="array">The array of values.</param>
    /// <returns>The mean (average) value.</returns>
    public static T HorizontalReduceMeanArray<T>(T[] array)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var sum = ops.Sum(array.AsSpan());
        return ops.Divide(sum, ops.FromDouble(array.Length));
    }

    #endregion

    #region Value Helpers

    /// <summary>
    /// Gets the minimum value for the numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <returns>The minimum representable value.</returns>
    public static T MinValue<T>()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.MinValue;
    }

    /// <summary>
    /// Gets the maximum value for the numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <returns>The maximum representable value.</returns>
    public static T MaxValue<T>()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.MaxValue;
    }

    /// <summary>
    /// Gets the zero value for the numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <returns>The zero value.</returns>
    public static T Zero<T>()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Zero;
    }

    #endregion

    #region Binary Operations (Vector-based)

    /// <summary>
    /// Performs element-wise addition on two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand vector.</param>
    /// <param name="right">The right operand vector.</param>
    /// <param name="result">The result vector.</param>
    public static void Add<T>(Vector<T> left, Vector<T> right, Vector<T> result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Add(left.AsSpan(), right.AsSpan(), result.AsWritableSpan());
    }

    /// <summary>
    /// Performs element-wise subtraction on two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand vector.</param>
    /// <param name="right">The right operand vector.</param>
    /// <param name="result">The result vector.</param>
    public static void Subtract<T>(Vector<T> left, Vector<T> right, Vector<T> result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Subtract(left.AsSpan(), right.AsSpan(), result.AsWritableSpan());
    }

    /// <summary>
    /// Performs element-wise multiplication on two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand vector.</param>
    /// <param name="right">The right operand vector.</param>
    /// <param name="result">The result vector.</param>
    public static void Multiply<T>(Vector<T> left, Vector<T> right, Vector<T> result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Multiply(left.AsSpan(), right.AsSpan(), result.AsWritableSpan());
    }

    /// <summary>
    /// Performs element-wise division on two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand vector.</param>
    /// <param name="right">The right operand vector.</param>
    /// <param name="result">The result vector.</param>
    public static void Divide<T>(Vector<T> left, Vector<T> right, Vector<T> result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Divide(left.AsSpan(), right.AsSpan(), result.AsWritableSpan());
    }

    #endregion

    #region Binary Operations (Tensor-based)

    /// <summary>
    /// Performs element-wise addition on two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand tensor.</param>
    /// <param name="right">The right operand tensor.</param>
    /// <param name="result">The result tensor.</param>
    public static void Add<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Add(left.AsSpan(), right.AsSpan(), result.AsWritableSpan());
    }

    /// <summary>
    /// Performs element-wise subtraction on two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand tensor.</param>
    /// <param name="right">The right operand tensor.</param>
    /// <param name="result">The result tensor.</param>
    public static void Subtract<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Subtract(left.AsSpan(), right.AsSpan(), result.AsWritableSpan());
    }

    /// <summary>
    /// Performs element-wise multiplication on two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand tensor.</param>
    /// <param name="right">The right operand tensor.</param>
    /// <param name="result">The result tensor.</param>
    public static void Multiply<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Multiply(left.AsSpan(), right.AsSpan(), result.AsWritableSpan());
    }

    /// <summary>
    /// Performs element-wise division on two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand tensor.</param>
    /// <param name="right">The right operand tensor.</param>
    /// <param name="result">The result tensor.</param>
    public static void Divide<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Divide(left.AsSpan(), right.AsSpan(), result.AsWritableSpan());
    }

    #endregion

    #region Binary Operations (Array-based for Expression Trees)

    /// <summary>
    /// Performs element-wise addition on two arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand array.</param>
    /// <param name="right">The right operand array.</param>
    /// <param name="result">The result array.</param>
    public static void AddArrays<T>(T[] left, T[] right, T[] result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Add(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    /// <summary>
    /// Performs element-wise subtraction on two arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand array.</param>
    /// <param name="right">The right operand array.</param>
    /// <param name="result">The result array.</param>
    public static void SubtractArrays<T>(T[] left, T[] right, T[] result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Subtract(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    /// <summary>
    /// Performs element-wise multiplication on two arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand array.</param>
    /// <param name="right">The right operand array.</param>
    /// <param name="result">The result array.</param>
    public static void MultiplyArrays<T>(T[] left, T[] right, T[] result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Multiply(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    /// <summary>
    /// Performs element-wise division on two arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The left operand array.</param>
    /// <param name="right">The right operand array.</param>
    /// <param name="result">The result array.</param>
    public static void DivideArrays<T>(T[] left, T[] right, T[] result)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Divide(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    #endregion

    #region Unary Operations (Vector-based)

    /// <summary>
    /// Applies ReLU activation to a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input vector.</param>
    /// <param name="output">The output vector.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> ReLU (Rectified Linear Unit) sets all negative values to zero
    /// and keeps positive values unchanged. It's the most common activation function in neural networks.
    ///
    /// For example: ReLU([-2, -1, 0, 1, 2]) = [0, 0, 0, 1, 2]
    /// </para>
    /// </remarks>
    public static void ApplyReLU<T>(Vector<T> input, Vector<T> output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var zero = ops.Zero;
        var inputSpan = input.AsSpan();
        var outputSpan = output.AsWritableSpan();
        for (int i = 0; i < inputSpan.Length; i++)
        {
            outputSpan[i] = ops.GreaterThan(inputSpan[i], zero) ? inputSpan[i] : zero;
        }
    }

    /// <summary>
    /// Applies ReLU gradient during backpropagation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient from the output.</param>
    /// <param name="forwardInput">The original input from the forward pass.</param>
    /// <param name="gradInput">The gradient to propagate to the input.</param>
    public static void ApplyReLUGrad<T>(Vector<T> gradOutput, Vector<T> forwardInput, Vector<T> gradInput)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var zero = ops.Zero;
        var gradOutSpan = gradOutput.AsSpan();
        var forwardSpan = forwardInput.AsSpan();
        var gradInSpan = gradInput.AsWritableSpan();
        for (int i = 0; i < gradOutSpan.Length; i++)
        {
            gradInSpan[i] = ops.GreaterThan(forwardSpan[i], zero) ? gradOutSpan[i] : zero;
        }
    }

    /// <summary>
    /// Applies sigmoid activation to a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input vector.</param>
    /// <param name="output">The output vector.</param>
    public static void ApplySigmoid<T>(Vector<T> input, Vector<T> output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Sigmoid(input.AsSpan(), output.AsWritableSpan());
    }

    /// <summary>
    /// Applies tanh activation to a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input vector.</param>
    /// <param name="output">The output vector.</param>
    public static void ApplyTanh<T>(Vector<T> input, Vector<T> output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Tanh(input.AsSpan(), output.AsWritableSpan());
    }

    /// <summary>
    /// Applies element-wise exponential function to a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input vector.</param>
    /// <param name="output">The output vector.</param>
    public static void ApplyExp<T>(Vector<T> input, Vector<T> output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Exp(input.AsSpan(), output.AsWritableSpan());
    }

    /// <summary>
    /// Applies element-wise natural logarithm to a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input vector.</param>
    /// <param name="output">The output vector.</param>
    public static void ApplyLog<T>(Vector<T> input, Vector<T> output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Log(input.AsSpan(), output.AsWritableSpan());
    }

    /// <summary>
    /// Applies softmax activation to a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input vector.</param>
    /// <param name="output">The output vector.</param>
    public static void ApplySoftMax<T>(Vector<T> input, Vector<T> output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.SoftMax(input.AsSpan(), output.AsWritableSpan());
    }

    #endregion

    #region Unary Operations (Array-based for Expression Trees)

    /// <summary>
    /// Applies ReLU activation to an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input array.</param>
    /// <param name="output">The output array.</param>
    public static void ApplyReLUArrays<T>(T[] input, T[] output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var zero = ops.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = ops.GreaterThan(input[i], zero) ? input[i] : zero;
        }
    }

    /// <summary>
    /// Applies sigmoid activation to an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input array.</param>
    /// <param name="output">The output array.</param>
    public static void ApplySigmoidArrays<T>(T[] input, T[] output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Sigmoid(input.AsSpan(), output.AsSpan());
    }

    /// <summary>
    /// Applies tanh activation to an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input array.</param>
    /// <param name="output">The output array.</param>
    public static void ApplyTanhArrays<T>(T[] input, T[] output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Tanh(input.AsSpan(), output.AsSpan());
    }

    /// <summary>
    /// Applies element-wise exponential function to an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input array.</param>
    /// <param name="output">The output array.</param>
    public static void ApplyExpArrays<T>(T[] input, T[] output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Exp(input.AsSpan(), output.AsSpan());
    }

    /// <summary>
    /// Applies element-wise natural logarithm to an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input array.</param>
    /// <param name="output">The output array.</param>
    public static void ApplyLogArrays<T>(T[] input, T[] output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.Log(input.AsSpan(), output.AsSpan());
    }

    /// <summary>
    /// Applies softmax activation to an array.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input array.</param>
    /// <param name="output">The output array.</param>
    public static void ApplySoftMaxArrays<T>(T[] input, T[] output)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        ops.SoftMax(input.AsSpan(), output.AsSpan());
    }

    #endregion

    #region Dot Product and Similarity

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The first vector.</param>
    /// <param name="right">The second vector.</param>
    /// <returns>The dot product.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The dot product multiplies corresponding elements
    /// and sums the results. For example: Dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 32.
    /// It's fundamental to neural network computations.
    /// </para>
    /// </remarks>
    public static T Dot<T>(Vector<T> left, Vector<T> right)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Dot(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Computes the dot product of two arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The first array.</param>
    /// <param name="right">The second array.</param>
    /// <returns>The dot product.</returns>
    public static T DotArray<T>(T[] left, T[] right)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Dot(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Computes the cosine similarity between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The first vector.</param>
    /// <param name="right">The second vector.</param>
    /// <returns>The cosine similarity (between -1 and 1).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cosine similarity measures how similar two vectors are
    /// based on the angle between them. A value of 1 means they point in the same direction,
    /// -1 means opposite directions, and 0 means they're perpendicular.
    /// </para>
    /// </remarks>
    public static T CosineSimilarity<T>(Vector<T> left, Vector<T> right)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.CosineSimilarity(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Computes the cosine similarity between two arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">The first array.</param>
    /// <param name="right">The second array.</param>
    /// <returns>The cosine similarity (between -1 and 1).</returns>
    public static T CosineSimilarityArray<T>(T[] left, T[] right)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.CosineSimilarity(left.AsSpan(), right.AsSpan());
    }

    #endregion
}
