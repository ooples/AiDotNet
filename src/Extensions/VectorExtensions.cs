using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for vector operations commonly used in AI and machine learning applications.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A vector in AI is simply a list of numbers that can represent data points, 
/// features, or weights in a model. These extension methods provide ways to manipulate and 
/// perform calculations on these lists of numbers.
/// </para>
/// </remarks>
public static class VectorExtensions
{
    /// <summary>
    /// Creates a new vector containing a subset of elements from the original vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The source vector to slice.</param>
    /// <param name="start">The zero-based index at which to begin the slice.</param>
    /// <param name="length">The number of elements to include in the slice.</param>
    /// <returns>A new vector containing the specified elements.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is like taking a portion of a list. For example, if you have 
    /// a vector [1, 2, 3, 4, 5] and you call Slice(1, 3), you'll get [2, 3, 4].
    /// </para>
    /// </remarks>
    public static Vector<T> Slice<T>(this Vector<T> vector, int start, int length)
    {
        var slicedVector = new Vector<T>(length);
        var numOps = MathHelper.GetNumericOperations<T>();
        var sourceSpan = vector.AsSpan().Slice(start, length);
        var destSpan = slicedVector.AsWritableSpan();
        numOps.Copy(sourceSpan, destSpan);

        return slicedVector;
    }

    /// <summary>
    /// Calculates the Euclidean norm (magnitude or length) of the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to calculate the norm for.</param>
    /// <returns>The Euclidean norm of the vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The norm is the "length" of a vector. For a 2D vector [x, y], 
    /// it's calculated as √(x² + y²), which is the same as the Pythagorean theorem. 
    /// For vectors with more dimensions, it's the square root of the sum of all squared elements.
    /// </para>
    /// </remarks>
    public static T Norm<T>(this Vector<T> vector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var span = vector.AsSpan();

        // Manually compute sum of squares using vectorized multiply + sum
        var temp = new Vector<T>(vector.Length);
        var tempSpan = temp.AsWritableSpan();
        numOps.Multiply(span, span, tempSpan);
        T sumOfSquares = numOps.Sum(tempSpan);

        return numOps.Sqrt(sumOfSquares);
    }

    /// <summary>
    /// Converts a collection of integer indices to a list of single-element vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type to convert the indices to.</typeparam>
    /// <param name="indices">The collection of integer indices to convert.</param>
    /// <returns>A list of vectors, each containing a single element representing an index.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a list of numbers (like [1, 2, 3]) and converts each number 
    /// into its own tiny vector. So [1, 2, 3] becomes [[1], [2], [3]], where each inner bracket is a vector.
    /// </para>
    /// </remarks>
    public static List<Vector<T>> ToVectorList<T>(this IEnumerable<int> indices)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return indices.Select(index => new Vector<T>(new[] { numOps.FromDouble(index) })).ToList();
    }

    /// <summary>
    /// Converts a collection of vectors to a list of integers.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vectors">The collection of vectors to convert.</param>
    /// <returns>A list of integers extracted from the vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This does the opposite of ToVectorList. It takes a list of vectors 
    /// (like [[1], [2], [3]]) and converts it to a simple list of numbers [1, 2, 3].
    /// </para>
    /// </remarks>
    public static List<int> ToIntList<T>(this IEnumerable<Vector<T>> vectors)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return vectors.SelectMany(v => v.Select(x => numOps.ToInt32(x))).ToList();
    }

    /// <summary>
    /// Creates a diagonal matrix from a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to create a diagonal matrix from.</param>
    /// <returns>A square matrix with the vector elements on the diagonal and zeros elsewhere.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A diagonal matrix has numbers only along its diagonal (top-left to bottom-right) 
    /// and zeros everywhere else. This method takes a vector like [1, 2, 3] and creates a matrix like:
    /// [1, 0, 0]
    /// [0, 2, 0]
    /// [0, 0, 3]
    /// </para>
    /// </remarks>
    public static Matrix<T> CreateDiagonal<T>(this Vector<T> vector)
    {
        var matrix = new Matrix<T>(vector.Length, vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            matrix[i, i] = vector[i];
        }

        return matrix;
    }

    /// <summary>
    /// Returns the indices that would sort the vector in ascending order.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to get the sorted indices for.</param>
    /// <returns>An array of indices that would sort the vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This doesn't actually sort the vector. Instead, it tells you the positions 
    /// (indices) in which the elements would appear if they were sorted. For example, if your vector is 
    /// [3, 1, 2], the result would be [1, 2, 0], meaning the smallest element is at position 1, 
    /// the second smallest at position 2, and the largest at position 0.
    /// </para>
    /// </remarks>
    public static int[] Argsort<T>(this Vector<T> vector)
    {
        return Enumerable.Range(0, vector.Length).OrderBy(i => vector[i]).ToArray();
    }

    /// <summary>
    /// Creates a new vector by repeating the original vector a specified number of times.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to repeat.</param>
    /// <param name="count">The number of times to repeat the vector.</param>
    /// <returns>A new vector containing the original vector repeated the specified number of times.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a longer vector by copying the original vector multiple times.
    /// For example, if your vector is [1, 2] and you repeat it 3 times, you'll get [1, 2, 1, 2, 1, 2].
    /// </para>
    /// </remarks>
    public static Vector<T> Repeat<T>(this Vector<T> vector, int count)
    {
        var result = new Vector<T>(vector.Length * count);
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < vector.Length; j++)
            {
                result[i * vector.Length + j] = vector[j];
            }
        }

        return result;
    }

    /// <summary>
    /// Adds two vectors element by element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="left">The first vector.</param>
    /// <param name="right">The second vector.</param>
    /// <returns>A new vector where each element is the sum of the corresponding elements in the input vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This adds two vectors together by adding their corresponding elements.
    /// For example, [1, 2, 3] + [4, 5, 6] = [5, 7, 9].
    /// </para>
    /// </remarks>
    public static Vector<T> Add<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have same dimension");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length);
        var leftSpan = left.AsSpan();
        var rightSpan = right.AsSpan();
        var destSpan = result.AsWritableSpan();
        numOps.Add(leftSpan, rightSpan, destSpan);

        return result;
    }

    /// <summary>
    /// Applies the exponential function to each element of the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to transform.</param>
    /// <returns>A new vector with the exponential function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The exponential function (e^x) is applied to each number in the vector.
    /// For example, if your vector is [0, 1, 2], the result would be approximately [1, 2.72, 7.39].
    /// This is commonly used in machine learning algorithms like softmax.
    /// </para>
    /// </remarks>
    public static VectorBase<T> PointwiseExp<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value =>
        {
            double doubleValue = Convert.ToDouble(value);
            double expValue = Math.Exp(doubleValue);

            return operations.FromDouble(expValue);
        });
    }

    /// <summary>
    /// Applies the natural logarithm function to each element of the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to transform.</param>
    /// <returns>A new vector with the natural logarithm applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The natural logarithm (ln) is the inverse of the exponential function.
    /// For example, if your vector is [1, 2.72, 7.39], the result would be approximately [0, 1, 2].
    /// This is commonly used in machine learning for transforming data and in algorithms like 
    /// cross-entropy loss calculation.
    /// </para>
    /// </remarks>
    public static VectorBase<T> PointwiseLog<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value =>
        {
            double doubleValue = Convert.ToDouble(value);
            double logValue = Math.Log(doubleValue);

            return operations.FromDouble(logValue);
        });
    }

    /// <summary>
    /// Subtracts the elements of the right vector from the corresponding elements of the left vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="left">The vector to subtract from (minuend).</param>
    /// <param name="right">The vector to subtract (subtrahend).</param>
    /// <returns>A new vector containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This performs element-by-element subtraction between two vectors.
    /// For example, if you have [5, 8, 3] and [2, 3, 1], the result will be [3, 5, 2].
    /// This is useful for calculating differences between data points or model predictions and actual values.
    /// </para>
    /// </remarks>
    public static Vector<T> Subtract<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have same dimension");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length);
        var leftSpan = left.AsSpan();
        var rightSpan = right.AsSpan();
        var destSpan = result.AsWritableSpan();
        numOps.Subtract(leftSpan, rightSpan, destSpan);

        return result;
    }

    /// <summary>
    /// Calculates the dot product (scalar product) of two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="left">The first vector.</param>
    /// <param name="right">The second vector.</param>
    /// <returns>The scalar result of the dot product operation.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The dot product multiplies corresponding elements of two vectors and then adds all the results.
    /// For example, the dot product of [1, 2, 3] and [4, 5, 6] is (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32.
    /// This is fundamental in machine learning for calculating similarities between vectors, projections, and in neural network operations.
    /// </para>
    /// </remarks>
    public static T DotProduct<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have same dimension");

        var numOps = MathHelper.GetNumericOperations<T>();
        var leftSpan = left.AsSpan();
        var rightSpan = right.AsSpan();
        return numOps.Dot(leftSpan, rightSpan);
    }

    /// <summary>
    /// Divides each element of the vector by a scalar value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector whose elements will be divided.</param>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <returns>A new vector with each element divided by the scalar.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This divides every number in your vector by the same value.
    /// For example, if your vector is [10, 20, 30] and you divide by 10, the result is [1, 2, 3].
    /// This is commonly used for normalization in machine learning, such as when calculating averages or scaling data.
    /// </para>
    /// </remarks>
    public static Vector<T> Divide<T>(this Vector<T> vector, T scalar)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        var sourceSpan = vector.AsSpan();
        var destSpan = result.AsWritableSpan();
        numOps.DivideScalar(sourceSpan, scalar, destSpan);

        return result;
    }

    /// <summary>
    /// Multiplies each element of the vector by a scalar value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector whose elements will be multiplied.</param>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <returns>A new vector with each element multiplied by the scalar.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This multiplies every number in your vector by the same value.
    /// For example, if your vector is [1, 2, 3] and you multiply by 10, the result is [10, 20, 30].
    /// This is commonly used for scaling in machine learning, such as when applying learning rates or weights.
    /// </para>
    /// </remarks>
    public static Vector<T> Multiply<T>(this Vector<T> vector, T scalar)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        var sourceSpan = vector.AsSpan();
        var destSpan = result.AsWritableSpan();
        numOps.MultiplyScalar(sourceSpan, scalar, destSpan);

        return result;
    }

    /// <summary>
    /// Multiplies a vector by a matrix, performing a vector-matrix multiplication.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector and matrix elements.</typeparam>
    /// <param name="vector">The vector to multiply (must have length equal to the number of matrix rows).</param>
    /// <param name="matrix">The matrix to multiply by.</param>
    /// <returns>A new vector resulting from the vector-matrix multiplication.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector length doesn't match the number of matrix rows.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vector-matrix multiplication transforms a vector using a matrix.
    /// The vector must have the same number of elements as the matrix has rows.
    /// This operation is fundamental in machine learning for transforming data, applying weights in neural networks,
    /// and implementing linear transformations.
    /// </para>
    /// <para>
    /// For example, multiplying a vector [1, 2, 3] by a 3×2 matrix [[1, 4], [2, 5], [3, 6]] 
    /// results in a vector [1×1 + 2×2 + 3×3, 1×4 + 2×5 + 3×6] = [14, 32].
    /// </para>
    /// </remarks>
    public static Vector<T> Multiply<T>(this Vector<T> vector, Matrix<T> matrix)
    {
        if (vector.Length != matrix.Rows)
            throw new ArgumentException("Vector length must match matrix rows");

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            var sum = operations.Zero;
            for (int i = 0; i < matrix.Rows; i++)
            {
                sum = operations.Add(sum, operations.Multiply(vector[i], matrix[i, j]));
            }

            result[j] = sum;
        }

        return result;
    }

    /// <summary>
    /// Multiplies corresponding elements of two vectors together (Hadamard product).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="left">The first vector.</param>
    /// <param name="right">The second vector.</param>
    /// <returns>A new vector containing the element-wise product.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This multiplies each element in the first vector by the corresponding element 
    /// in the second vector. For example, if you have [1, 2, 3] and [4, 5, 6], the result will be [4, 10, 18].
    /// </para>
    /// <para>
    /// This operation (also called the Hadamard product) is commonly used in neural networks, 
    /// particularly when applying masks, attention mechanisms, or element-wise gates in recurrent networks.
    /// </para>
    /// </remarks>
    public static Vector<T> PointwiseMultiply<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have the same length");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length);
        var leftSpan = left.AsSpan();
        var rightSpan = right.AsSpan();
        var destSpan = result.AsWritableSpan();
        numOps.Multiply(leftSpan, rightSpan, destSpan);

        return result;
    }

    /// <summary>
    /// Computes the outer product of two vectors, resulting in a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="leftVector">The left vector in the outer product.</param>
    /// <param name="rightVector">The right vector in the outer product.</param>
    /// <returns>A matrix where each element (i,j) is the product of leftVector[i] and rightVector[j].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The outer product creates a matrix by multiplying each element of the first vector
    /// with each element of the second vector. If you have a vector [a,b,c] and another vector [x,y],
    /// the result is a 3×2 matrix:
    /// [a*x, a*y]
    /// [b*x, b*y]
    /// [c*x, c*y]
    /// This operation is useful in neural networks for creating weight matrices.
    /// </para>
    /// </remarks>
    public static Matrix<T> OuterProduct<T>(this Vector<T> leftVector, Vector<T> rightVector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var matrix = new Matrix<T>(leftVector.Length, rightVector.Length);

        for (int i = 0; i < leftVector.Length; i++)
        {
            for (int j = 0; j < rightVector.Length; j++)
            {
                matrix[i, j] = operations.Multiply(leftVector[i], rightVector[j]);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Calculates the magnitude (length) of a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector whose magnitude is to be calculated.</param>
    /// <returns>The magnitude of the vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The magnitude is the "length" of a vector, calculated using the Pythagorean theorem.
    /// For a vector [a,b,c], the magnitude is √(a² + b² + c²). This is useful for normalizing vectors
    /// or measuring distances in machine learning algorithms.
    /// </para>
    /// </remarks>
    public static T Magnitude<T>(this Vector<T> vector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var span = vector.AsSpan();

        // Manually compute sum of squares using vectorized multiply + sum
        var temp = new Vector<T>(vector.Length);
        var tempSpan = temp.AsWritableSpan();
        numOps.Multiply(span, span, tempSpan);
        T sumOfSquares = numOps.Sum(tempSpan);

        return numOps.Sqrt(sumOfSquares);
    }

    /// <summary>
    /// Divides each element of the left vector by the corresponding element of the right vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="left">The vector whose elements will be divided (dividend).</param>
    /// <param name="right">The vector containing the divisors.</param>
    /// <returns>A new vector with each element being the result of left[i] / right[i].</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This performs element-by-element division between two vectors.
    /// For example, dividing [10,8,6] by [2,4,3] results in [5,2,2].
    /// This is commonly used in data normalization and feature scaling.
    /// </para>
    /// </remarks>
    public static Vector<T> PointwiseDivide<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have the same length");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length);
        var leftSpan = left.AsSpan();
        var rightSpan = right.AsSpan();
        var destSpan = result.AsWritableSpan();
        numOps.Divide(leftSpan, rightSpan, destSpan);

        return result;
    }

    /// <summary>
    /// Multiplies each element of the left vector by the corresponding element of the right vector and stores the result in the left vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="left">The vector to be modified with the multiplication results.</param>
    /// <param name="right">The vector containing the multipliers.</param>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method performs element-by-element multiplication and modifies the original vector.
    /// For example, if left is [1,2,3] and right is [4,5,6], after this operation left becomes [4,10,18].
    /// The "InPlace" in the name indicates that the operation modifies the original vector rather than creating a new one.
    /// </para>
    /// </remarks>
    public static void PointwiseMultiplyInPlace<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have the same length");

        var numOps = MathHelper.GetNumericOperations<T>();
        var leftSpan = left.AsWritableSpan();
        var rightSpan = right.AsSpan();
        numOps.Multiply(leftSpan, rightSpan, leftSpan);
    }

    /// <summary>
    /// Finds the maximum value in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to search.</param>
    /// <returns>The maximum value in the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the largest number in a vector.
    /// For example, in the vector [3,7,2,5], the maximum value is 7.
    /// This is commonly used in algorithms like gradient descent to check convergence
    /// or in classification to find the most likely class.
    /// </para>
    /// </remarks>
    public static T Max<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var numOps = MathHelper.GetNumericOperations<T>();
        var span = vector.AsSpan();
        return numOps.Max(span);
    }

    /// <summary>
    /// Calculates the average (mean) of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector whose elements will be averaged.</param>
    /// <returns>The average value of all elements in the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The average is calculated by summing all elements and dividing by the count.
    /// For example, the average of [2,4,6,8] is (2+4+6+8)/4 = 5.
    /// Averages are fundamental in statistics and machine learning for summarizing data.
    /// </para>
    /// </remarks>
    public static T Average<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var operations = MathHelper.GetNumericOperations<T>();
        T sum = vector.Sum();
        return operations.Divide(sum, operations.FromDouble(vector.Length));
    }

    /// <summary>
    /// Finds the minimum value in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to search.</param>
    /// <returns>The minimum value in the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the smallest number in a vector.
    /// For example, in the vector [3,7,2,5], the minimum value is 2.
    /// Finding minimums is useful in optimization algorithms and for identifying lower bounds in data.
    /// </para>
    /// </remarks>
    public static T Min<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var numOps = MathHelper.GetNumericOperations<T>();
        var span = vector.AsSpan();
        return numOps.Min(span);
    }

    /// <summary>
    /// Returns a new vector where each element is the sign of the corresponding element in the input vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector whose signs will be determined.</param>
    /// <returns>A new vector where each element is 1 for positive values, -1 for negative values, and 0 for zero.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sign function returns:
    /// - 1 if the number is positive
    /// - -1 if the number is negative
    /// - 0 if the number is zero
    /// 
    /// For example, applying this to [-5,0,3] results in [-1,0,1].
    /// This is useful in algorithms that need to know only the direction of values, not their magnitude.
    /// </para>
    /// </remarks>
    public static VectorBase<T> PointwiseSign<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value =>
        {
            if (operations.GreaterThan(value, operations.Zero))
                return operations.One;
            else if (operations.LessThan(value, operations.Zero))
                return operations.Negate(operations.One);
            else
                return operations.Zero;
        });
    }

    /// <summary>
    /// Finds the element with the largest absolute value in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to search.</param>
    /// <returns>The largest absolute value in the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method ignores the sign of numbers and finds the largest value by magnitude.
    /// For example, in the vector [-10, 5, 3], the absolute maximum is 10 (from -10).
    /// This is useful in algorithms where the size of values matters more than their direction.
    /// </para>
    /// </remarks>
    public static T AbsoluteMaximum<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var numOps = MathHelper.GetNumericOperations<T>();
        var span = vector.AsSpan();
        var temp = new Vector<T>(vector.Length);
        var tempSpan = temp.AsWritableSpan();
        numOps.Abs(span, tempSpan);
        return numOps.Max(tempSpan);
    }

    /// <summary>
    /// Finds the minimum element value in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to search.</param>
    /// <returns>The smallest value in the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the smallest number in your vector.
    /// For example, in the vector [5, 2, 8, 1], the minimum is 1.
    /// </para>
    /// </remarks>
    public static T Minimum<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var numOps = MathHelper.GetNumericOperations<T>();
        var span = vector.AsSpan();
        return numOps.Min(span);
    }

    /// <summary>
    /// Creates a new vector where each element is the absolute value of the corresponding element in the input vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector whose absolute values will be calculated.</param>
    /// <returns>A new vector containing the absolute values of the original vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The absolute value of a number is its distance from zero, ignoring the sign.
    /// For example, applying this to [-5, 0, 3] results in [5, 0, 3].
    /// This is useful when you need to work with magnitudes regardless of direction.
    /// </para>
    /// </remarks>
    public static VectorBase<T> PointwiseAbs<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.Abs(value));
    }

    /// <summary>
    /// Creates a new vector where each element is the square root of the corresponding element in the input vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector whose square roots will be calculated.</param>
    /// <returns>A new vector containing the square roots of the original vector elements.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The square root of a number is a value that, when multiplied by itself, gives the original number.
    /// For example, applying this to [4, 9, 16] results in [2, 3, 4].
    /// Square roots are commonly used in distance calculations and statistical analysis.
    /// </para>
    /// </remarks>
    public static VectorBase<T> PointwiseSqrt<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value =>
        {
            double doubleValue = Convert.ToDouble(value);
            double sqrtValue = Math.Sqrt(doubleValue);
            return operations.FromDouble(sqrtValue);
        });
    }

    /// <summary>
    /// Adds a scalar value to each element of the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to which the scalar will be added.</param>
    /// <param name="scalar">The scalar value to add to each element.</param>
    /// <returns>A new vector with the scalar added to each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method adds the same number to every element in your vector.
    /// For example, adding 5 to [1, 2, 3] results in [6, 7, 8].
    /// This is useful for operations like shifting data or applying offsets.
    /// </para>
    /// </remarks>
    public static Vector<T> Add<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.Add(value, scalar));
    }

    /// <summary>
    /// Creates a new vector where each element is the maximum of the corresponding element in the input vector and a scalar value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <param name="scalar">The scalar value to compare against.</param>
    /// <returns>A new vector where each element is the maximum of the original element and the scalar.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method compares each element in your vector with a given number,
    /// and keeps whichever is larger. For example, finding the maximum of each element in [1, 5, 3]
    /// compared to 2 results in [2, 5, 3].
    /// This is useful for setting minimum thresholds in data.
    /// </para>
    /// </remarks>
    public static VectorBase<T> Maximum<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.GreaterThan(value, scalar) ? value : scalar);
    }

    /// <summary>
    /// Calculates the sum of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector whose elements will be summed.</param>
    /// <returns>The sum of all elements in the vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method adds up all the numbers in your vector.
    /// For example, the sum of [1, 2, 3, 4] is 10.
    /// Summing is a fundamental operation used in many algorithms, including calculating averages and totals.
    /// </para>
    /// </remarks>
    public static T Sum<T>(this Vector<T> vector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var span = vector.AsSpan();
        return numOps.Sum(span);
    }

    /// <summary>
    /// Applies a function to each element of the vector and returns a new vector with the results.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to transform.</param>
    /// <param name="function">The function to apply to each element.</param>
    /// <returns>A new vector containing the transformed elements.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you apply any operation to each element in your vector.
    /// For example, you could multiply each element by 2, or add 5 to each element.
    /// This is a powerful tool that allows you to customize how you process your data.
    /// </para>
    /// </remarks>
    public static Vector<T> Transform<T>(this Vector<T> vector, Func<T, T> function)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = function(vector[i]);
        }

        return result;
    }

    /// <summary>
    /// Finds the index of the maximum value in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to search.</param>
    /// <returns>The index of the maximum value in the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the position of the largest number in your vector.
    /// For example, in the vector [3, 7, 2, 5], the maximum value 7 is at index 1 (positions are counted starting from 0).
    /// This is useful when you need to know where the highest value is, not just what it is.
    /// </para>
    /// </remarks>
    public static int MaxIndex<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        int maxIndex = 0;
        T max = vector[0];
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 1; i < vector.Length; i++)
        {
            if (ops.GreaterThan(vector[i], max))
            {
                max = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Finds the index of the minimum value in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to search.</param>
    /// <returns>The index of the minimum value in the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the position of the smallest number in your vector.
    /// For example, in the vector [3, 7, 2, 5], the minimum value 2 is at index 2 (positions are counted starting from 0).
    /// This is useful when you need to know where the lowest value is, not just what it is.
    /// </para>
    /// </remarks>
    public static int MinIndex<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        int minIndex = 0;
        T min = vector[0];
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 1; i < vector.Length; i++)
        {
            if (ops.LessThan(vector[i], min))
            {
                min = vector[i];
                minIndex = i;
            }
        }

        return minIndex;
    }

    /// <summary>
    /// Creates a new vector containing a subset of elements from the original vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The source vector.</param>
    /// <param name="startIndex">The zero-based index at which to start extracting elements.</param>
    /// <param name="count">The number of elements to extract.</param>
    /// <returns>A new vector containing the specified subset of elements.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when startIndex is negative or when the range exceeds the vector's length.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you take a "slice" of your vector. For example,
    /// if you have a vector [5, 10, 15, 20, 25] and you call SubVector(1, 3), you'll get
    /// a new vector with [10, 15, 20]. The first parameter (1) is where to start, and the
    /// second parameter (3) is how many elements to include.
    /// </para>
    /// </remarks>
    public static Vector<T> SubVector<T>(this Vector<T> vector, int startIndex, int count)
    {
        if (startIndex < 0 || startIndex + count > vector.Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex));

        var result = new Vector<T>(count);
        var numOps = MathHelper.GetNumericOperations<T>();
        var sourceSpan = vector.AsSpan().Slice(startIndex, count);
        var destSpan = result.AsWritableSpan();
        numOps.Copy(sourceSpan, destSpan);

        return result;
    }

    /// <summary>
    /// Creates a new vector by selecting elements from the original vector at specified indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The source vector.</param>
    /// <param name="indices">An array of indices specifying which elements to select.</param>
    /// <returns>A new vector containing the elements at the specified indices.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the indices array is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when any index in the array is outside the bounds of the vector.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you pick specific elements from your vector by their positions.
    /// For example, if you have a vector [5, 10, 15, 20, 25] and you call SubVector([0, 3, 4]), you'll get
    /// a new vector with [5, 20, 25]. The array [0, 3, 4] specifies which positions to select.
    /// </para>
    /// </remarks>
    public static Vector<T> SubVector<T>(this Vector<T> vector, int[] indices)
    {
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        var result = new Vector<T>(indices.Length);
        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] >= vector.Length)
                throw new ArgumentOutOfRangeException(nameof(indices), "Index out of range");

            result[i] = vector[indices[i]];
        }

        return result;
    }

    /// <summary>
    /// Applies a function to each element of the vector and returns a new vector with the results,
    /// allowing for a change in the element type.
    /// </summary>
    /// <typeparam name="T">The numeric type of the input vector elements.</typeparam>
    /// <typeparam name="TResult">The numeric type of the output vector elements.</typeparam>
    /// <param name="vector">The vector to transform.</param>
    /// <param name="function">The function to apply to each element.</param>
    /// <returns>A new vector containing the transformed elements.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is similar to the Transform method, but it allows you to change
    /// the type of your data. For example, you could convert a vector of integers to a vector of doubles,
    /// or apply any other type conversion to each element.
    /// </para>
    /// </remarks>
    public static VectorBase<TResult> Transform<T, TResult>(this Vector<T> vector, Func<T, TResult> function)
    {
        return vector.Transform(function);
    }

    /// <summary>
    /// Converts the vector to a diagonal matrix.
    /// </summary>
    /// <typeparam name="T">The type of the vector elements.</typeparam>
    /// <param name="vector">The vector to convert.</param>
    /// <returns>A square matrix with the vector elements on the diagonal and zeros elsewhere.</returns>
    /// <exception cref="ArgumentNullException">Thrown if the input vector is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A diagonal matrix is a special type of square matrix where all elements
    /// are zero except those on the main diagonal (top-left to bottom-right). This method creates
    /// such a matrix using your vector's values for the diagonal elements.
    /// </para>
    /// <para>
    /// For example, the vector [3, 5, 7] would become the matrix:
    /// [3, 0, 0]
    /// [0, 5, 0]
    /// [0, 0, 7]
    /// </para>
    /// <para>
    /// Diagonal matrices have special properties that make them useful in many mathematical operations
    /// and algorithms.
    /// </para>
    /// </remarks>
    public static Matrix<T> ToDiagonalMatrix<T>(this Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        int n = vector.Length;
        var matrix = Matrix<T>.CreateMatrix<T>(n, n);
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[i, j] = i == j ? vector[i] : ops.Zero;
            }
        }

        return matrix;
    }

    /// <summary>
    /// Converts the vector to a row matrix (1 x n).
    /// </summary>
    /// <typeparam name="T">The type of the vector elements.</typeparam>
    /// <param name="vector">The vector to convert.</param>
    /// <returns>A matrix with one row containing the vector elements.</returns>
    /// <exception cref="ArgumentNullException">Thrown if the input vector is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts your vector into a matrix with just one row.
    /// For example, the vector [3, 5, 7] would become the matrix:
    /// [3, 5, 7]
    /// </para>
    /// <para>
    /// Row matrices are useful in matrix multiplication operations and when you need to represent
    /// your data in matrix form for certain algorithms.
    /// </para>
    /// </remarks>
    public static Matrix<T> ToRowMatrix<T>(this Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        int n = vector.Length;
        var matrix = Matrix<T>.CreateMatrix<T>(1, n);
        for (int i = 0; i < n; ++i)
        {
            matrix[0, i] = vector[i];
        }

        return matrix;
    }

    /// <summary>
    /// Subtracts a scalar value from each element of the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector from which to subtract the scalar.</param>
    /// <param name="scalar">The scalar value to subtract from each element.</param>
    /// <returns>A new vector with the scalar subtracted from each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method subtracts the same number from every element in your vector.
    /// For example, if you have a vector [10, 15, 20] and subtract 5, you'll get [5, 10, 15].
    /// </para>
    /// <para>
    /// This is useful for operations like normalizing data or adjusting values by a constant amount.
    /// </para>
    /// </remarks>
    public static Vector<T> Subtract<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.Subtract(value, scalar));
    }

    /// <summary>
    /// Creates a new vector containing the first specified number of elements from the original vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The source vector.</param>
    /// <param name="length">The number of elements to extract from the beginning of the vector.</param>
    /// <returns>A new vector containing the first specified number of elements.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a shorter version of your vector by taking only
    /// the first few elements. For example, if you have a vector [5, 10, 15, 20, 25] and call
    /// Extract(3), you'll get a new vector with just [5, 10, 15].
    /// </para>
    /// <para>
    /// This is useful when you only need to work with the beginning portion of your data.
    /// </para>
    /// </remarks>
    public static Vector<T> Extract<T>(this Vector<T> vector, int length)
    {
        var result = new Vector<T>(length);
        var numOps = MathHelper.GetNumericOperations<T>();
        var sourceSpan = vector.AsSpan().Slice(0, length);
        var destSpan = result.AsWritableSpan();
        numOps.Copy(sourceSpan, destSpan);

        return result;
    }

    /// <summary>
    /// Extracts the real parts from a vector of complex numbers.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector of complex numbers.</param>
    /// <returns>A new vector containing only the real parts of the complex numbers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Complex numbers have two parts: a real part and an imaginary part.
    /// This method takes a vector of complex numbers and creates a new vector containing only
    /// the real parts.
    /// </para>
    /// <para>
    /// For example, if you have a vector of complex numbers [(3+2i), (5-1i), (7+0i)], this method
    /// would return [3, 5, 7].
    /// </para>
    /// <para>
    /// This is useful in signal processing and other fields where complex numbers are used but
    /// you need to work with just the real components.
    /// </para>
    /// </remarks>
    public static Vector<T> ToRealVector<T>(this Vector<Complex<T>> vector)
    {
        var count = vector.Length;
        var realVector = new Vector<T>(count);

        for (int i = 0; i < count; i++)
        {
            realVector[i] = vector[i].Real;
        }

        return realVector;
    }


    /// <summary>
    /// Reshapes a vector into a two-dimensional matrix with the specified number of rows and columns.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to reshape.</param>
    /// <param name="rows">The number of rows in the resulting matrix.</param>
    /// <param name="columns">The number of columns in the resulting matrix.</param>
    /// <returns>A matrix with the specified dimensions containing the vector elements.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the vector is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when rows or columns are less than or equal to zero, or when their product
    /// doesn't equal the vector length.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts your one-dimensional list of numbers (vector) into a 
    /// two-dimensional grid (matrix). Think of it like rearranging a line of items into rows and columns.
    /// </para>
    /// <para>
    /// For example, if you have a vector [1, 2, 3, 4, 5, 6] and reshape it to 2 rows and 3 columns,
    /// you'll get a matrix that looks like:
    /// [1, 2, 3]
    /// [4, 5, 6]
    /// </para>
    /// <para>
    /// The total number of elements must stay the same (rows × columns = vector length).
    /// </para>
    /// </remarks>
    public static Matrix<T> Reshape<T>(this Vector<T> vector, int rows, int columns)
    {
        if (vector == null)
        {
            throw new ArgumentNullException(nameof(vector), $"{nameof(vector)} can't be null");
        }

        if (rows <= 0)
        {
            throw new ArgumentException($"{nameof(rows)} needs to be an integer larger than 0", nameof(rows));
        }

        if (columns <= 0)
        {
            throw new ArgumentException($"{nameof(columns)} needs to be an integer larger than 0", nameof(columns));
        }

        var length = vector.Length;
        if (rows * columns != length)
        {
            throw new ArgumentException($"{nameof(rows)} and {nameof(columns)} multiplied together needs to be equal the array length");
        }

        var matrix = Matrix<T>.CreateMatrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = vector[i * columns + j];
            }
        }

        return matrix;
    }

    /// <summary>
    /// Converts the vector to a column matrix (n x 1).
    /// </summary>
    /// <typeparam name="T">The type of the vector elements.</typeparam>
    /// <param name="vector">The vector to convert.</param>
    /// <returns>A matrix with one column containing the vector elements.</returns>
    /// <exception cref="ArgumentNullException">Thrown if the input vector is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts your vector into a matrix with just one column.
    /// For example, the vector [3, 5, 7] would become the matrix:
    /// [3]
    /// [5]
    /// [7]
    /// </para>
    /// <para>
    /// Column matrices are useful in matrix multiplication operations and when you need to represent
    /// your data in matrix form for certain algorithms. In machine learning, input features are often
    /// represented as column matrices.
    /// </para>
    /// </remarks>
    public static Matrix<T> ToColumnMatrix<T>(this Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        int n = vector.Length;
        var matrix = Matrix<T>.CreateMatrix<T>(n, 1);
        for (int i = 0; i < n; ++i)
        {
            matrix[i, 0] = vector[i];
        }

        return matrix;
    }

    /// <summary>
    /// Calculates the standard deviation of the elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to calculate the standard deviation for.</param>
    /// <returns>The standard deviation of the vector elements.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Standard deviation measures how spread out the numbers in your vector are.
    /// A low standard deviation means the values tend to be close to the average (mean),
    /// while a high standard deviation means the values are spread out over a wider range.
    /// </para>
    /// <para>
    /// For example, the vector [2, 4, 4, 4, 5, 5, 7, 9] has a mean of 5 and a standard deviation of 2.
    /// Most values are within 2 units of the mean.
    /// </para>
    /// <para>
    /// This method uses the "sample standard deviation" formula, which divides by (n-1) instead of n.
    /// This is the standard approach in statistics when working with a sample of data rather than
    /// the entire population.
    /// </para>
    /// </remarks>
    public static T StandardDeviation<T>(this Vector<T> vector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = vector.Length;

        if (n < 2)
        {
            return numOps.Zero;
        }

        T mean = vector.Average();
        T sum = numOps.Zero;

        for (int i = 0; i < n; i++)
        {
            T diff = numOps.Subtract(vector[i], mean);
            sum = numOps.Add(sum, numOps.Multiply(diff, diff));
        }

        T variance = numOps.Divide(sum, numOps.FromDouble(n - 1));
        return numOps.Sqrt(variance);
    }

    /// <summary>
    /// Calculates the median value of the elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to calculate the median for.</param>
    /// <returns>The median value of the vector elements.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is null or empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The median is the middle value when all numbers are arranged in order.
    /// It's often used instead of the average (mean) when there are outliers or extreme values
    /// that might skew the mean.
    /// </para>
    /// <para>
    /// For example, in the vector [1, 3, 5, 7, 9], the median is 5.
    /// </para>
    /// <para>
    /// If there's an even number of elements, the median is the average of the two middle values.
    /// For example, in [1, 3, 5, 7], the median is (3 + 5) / 2 = 4.
    /// </para>
    /// </remarks>
    public static T Median<T>(this Vector<T> vector)
    {
        if (vector == null || vector.Length == 0)
            throw new ArgumentException("Vector is null or empty.");

        var sortedCopy = vector.ToArray();
        Array.Sort(sortedCopy);

        int mid = sortedCopy.Length / 2;
        if (sortedCopy.Length % 2 == 0)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            return numOps.Divide(numOps.Add(sortedCopy[mid - 1], sortedCopy[mid]), numOps.FromDouble(2.0));
        }
        else
        {
            return sortedCopy[mid];
        }
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>The Euclidean distance between the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Euclidean distance is the "straight-line" distance between two points
    /// in space. If you think of each vector as coordinates for a point, this method calculates
    /// how far apart those points are.
    /// </para>
    /// <para>
    /// For example, in 2D space, the Euclidean distance between points (1,2) and (4,6) would be:
    /// √[(4-1)² + (6-2)²] = √[9 + 16] = v25 = 5
    /// </para>
    /// <para>
    /// This concept extends to any number of dimensions. In machine learning, Euclidean distance
    /// is often used to measure similarity between data points or as part of clustering algorithms
    /// like K-means.
    /// </para>
    /// </remarks>
    public static T EuclideanDistance<T>(this Vector<T> v1, Vector<T> v2)
    {
        if (v1.Length != v2.Length)
        {
            throw new ArgumentException("Vectors must have the same length");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        T sumOfSquares = numOps.Zero;

        for (int i = 0; i < v1.Length; i++)
        {
            T diff = numOps.Subtract(v1[i], v2[i]);
            sumOfSquares = numOps.Add(sumOfSquares, numOps.Square(diff));
        }

        return numOps.Sqrt(sumOfSquares);
    }

    /// <summary>
    /// Creates a subvector from the given vector using the specified indices.
    /// </summary>
    /// <typeparam name="T">The type of elements in the vector.</typeparam>
    /// <param name="vector">The source vector.</param>
    /// <param name="indices">The indices of elements to include in the subvector.</param>
    /// <returns>A new vector containing only the specified elements.</returns>
    public static Vector<T> Subvector<T>(this Vector<T> vector, int[] indices)
    {
        var result = new Vector<T>(indices.Length);
        for (int i = 0; i < indices.Length; i++)
        {
            result[i] = vector[indices[i]];
        }

        return result;
    }
}
