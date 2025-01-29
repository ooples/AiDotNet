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

    public static Matrix<T> OuterProduct<T>(this Vector<T> vector1, Vector<T> vector2)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(vector1.Length, vector2.Length);
        for (int i = 0; i < vector1.Length; i++)
        {
            for (int j = 0; j < vector2.Length; j++)
            {
                result[i, j] = numOps.Multiply(vector1[i], vector2[j]);
            }
        }

        return result;
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