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
}