namespace AiDotNet.Extensions;

public static class VectorExtensions
{
    public static Vector<T> Slice<T>(this Vector<T> vector, int start, int length)
    {
        var slicedVector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            slicedVector[i] = vector[start + i];
        }

        return slicedVector;
    }

    public static T Norm<T>(this Vector<T> vector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;
        int n = vector.Length;
        for (int i = 0; i < n; i++)
        {
            sum = numOps.Add(sum, numOps.Multiply(vector[i], vector[i]));
        }

        return numOps.Sqrt(sum);
    }

    public static List<Vector<T>> ToVectorList<T>(this IEnumerable<int> indices)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return indices.Select(index => new Vector<T>(new[] { numOps.FromDouble(index) })).ToList();
    }

    public static List<int> ToIntList<T>(this IEnumerable<Vector<T>> vectors)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return vectors.SelectMany(v => v.Select(x => numOps.ToInt32(x))).ToList();
    }

    public static Matrix<T> CreateDiagonal<T>(this Vector<T> vector)
    {
        var matrix = new Matrix<T>(vector.Length, vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            matrix[i, i] = vector[i];
        }

        return matrix;
    }

    public static int[] Argsort<T>(this Vector<T> vector)
    {
        return [.. Enumerable.Range(0, vector.Length).OrderBy(i => vector[i])];
    }

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

    public static Vector<T> Add<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have same dimension");

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length);
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = operations.Add(left[i], right[i]);
        }

        return result;
    }

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

    public static Vector<T> Subtract<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have same dimension");

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length);

        for (int i = 0; i < left.Length; i++)
        {
            result[i] = operations.Subtract(left[i], right[i]);
        }

        return result;
    }

    public static T DotProduct<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have same dimension");

        var operations = MathHelper.GetNumericOperations<T>();
        var sum = operations.Zero;
        for (int i = 0; i < left.Length; i++)
        {
            sum = operations.Add(sum, operations.Multiply(left[i], right[i]));
        }

        return sum;
    }

    public static Vector<T> Divide<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = operations.Divide(vector[i], scalar);
        }

        return result;
    }

    public static Vector<T> Multiply<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = operations.Multiply(vector[i], scalar);
        }

        return result;
    }

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

    public static Vector<T> PointwiseMultiply<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have the same length");

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length);
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = operations.Multiply(left[i], right[i]);
        }

        return result;
    }

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

    public static T Magnitude<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        T sum = operations.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sum = operations.Add(sum, operations.Multiply(vector[i], vector[i]));
        }

        return operations.Sqrt(sum);
    }

    public static Vector<T> PointwiseDivide<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have the same length");

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length);
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = operations.Divide(left[i], right[i]);
        }

        return result;
    }

    public static void PointwiseMultiplyInPlace<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have the same length");

        var operations = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < left.Length; i++)
        {
            left[i] = operations.Multiply(left[i], right[i]);
        }
    }

    public static T Max<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var operations = MathHelper.GetNumericOperations<T>();
        T max = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (operations.GreaterThan(vector[i], max))
                max = vector[i];
        }

        return max;
    }

    public static T Average<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var operations = MathHelper.GetNumericOperations<T>();
        T sum = vector.Sum();
        return operations.Divide(sum, operations.FromDouble(vector.Length));
    }

    public static T Min<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var operations = MathHelper.GetNumericOperations<T>();
        T min = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (operations.LessThan(vector[i], min))
                min = vector[i];
        }

        return min;
    }

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

    public static T AbsoluteMaximum<T>(this Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var operations = MathHelper.GetNumericOperations<T>();
        T maxAbs = operations.Abs(vector[0]);
        for (int i = 1; i < vector.Length; i++)
        {
            T absValue = operations.Abs(vector[i]);
            if (operations.GreaterThan(absValue, maxAbs))
                maxAbs = absValue;
        }

        return maxAbs;
    }

    public static VectorBase<T> PointwiseAbs<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.Abs(value));
    }

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

    public static Vector<T> Add<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.Add(value, scalar));
    }

    public static VectorBase<T> Maximum<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.GreaterThan(value, scalar) ? value : scalar);
    }

    public static T Sum<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        T sum = operations.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sum = operations.Add(sum, vector[i]);
        }

        return sum;
    }

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

    public static Vector<T> SubVector<T>(this Vector<T> vector, int startIndex, int count)
    {
        if (startIndex < 0 || startIndex + count > vector.Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex));

        var result = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            result[i] = vector[startIndex + i];
        }

        return result;
    }

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
    public static Matrix<T> ToRowMatrix<T>(this Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        int n = vector.Length;
        var matrix = Matrix<T>.CreateMatrix<T>(n, 1);
        for (int i = 0; i < n; ++i)
        {
            matrix[0, i] = vector[i];
        }
        
        return matrix;
    }

    public static Vector<T> Subtract<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.Subtract(value, scalar));
    }

    public static Vector<T> Extract<T>(this Vector<T> vector, int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = vector[i];
        }

        return result;
    }

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
}