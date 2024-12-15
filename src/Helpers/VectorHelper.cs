namespace AiDotNet.Helpers;

public static class VectorHelper
{
    public static Vector<T> Add<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have same dimension");

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length, operations);
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
        var result = new Vector<T>(left.Length, operations);

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
        var result = new Vector<T>(vector.Length, operations);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = operations.Divide(vector[i], scalar);
        }

        return result;
    }

    public static Vector<T> Multiply<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length, operations);
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
        var result = new Vector<T>(matrix.Columns, operations);
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

    public static Vector<T> CreateVector<T>(int size)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return new Vector<T>(size, operations);
    }

    public static Matrix<T> OuterProduct<T>(this Vector<T> leftVector, Vector<T> rightVector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var matrix = new Matrix<T>(leftVector.Length, rightVector.Length, operations);

        for (int i = 0; i < leftVector.Length; i++)
        {
            for (int j = 0; j < rightVector.Length; j++)
            {
                matrix[i, j] = operations.Multiply(leftVector[i], rightVector[j]);
            }
        }

        return matrix;
    }

    public static Vector<T> PointwiseMultiply<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have the same length");

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length, operations);
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = operations.Multiply(left[i], right[i]);
        }

        return result;
    }

    public static Vector<T> PointwiseDivide<T>(this Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must have the same length");

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(left.Length, operations);
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

    public static Vector<T> Subtract<T>(this Vector<T> vector, T scalar)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        return vector.Transform(value => operations.Subtract(value, scalar));
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
        var result = new Vector<T>(vector.Length, operations);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = function(vector[i]);
        }

        return result;
    }

    public static int MaxIndex<T>(this Vector<T> vector) where T : IComparable<T>
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        int maxIndex = 0;
        T max = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (vector[i].CompareTo(max) > 0)
            {
                max = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public static int MinIndex<T>(this Vector<T> vector) where T : IComparable<T>
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        int minIndex = 0;
        T min = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (vector[i].CompareTo(min) < 0)
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

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(count, operations);
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

        var operations = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(indices.Length, operations);
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

    public static Matrix<T> ToRowMatrix<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var matrix = new Matrix<T>(1, vector.Length, operations);
        for (int i = 0; i < vector.Length; i++)
        {
            matrix[0, i] = vector[i];
        }

        return matrix;
    }

    public static Matrix<T> ToColumnMatrix<T>(this Vector<T> vector)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var matrix = new Matrix<T>(vector.Length, 1, operations);
        for (int i = 0; i < vector.Length; i++)
        {
            matrix[i, 0] = vector[i];
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
}