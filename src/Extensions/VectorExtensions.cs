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
}