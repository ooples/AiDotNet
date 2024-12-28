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
}