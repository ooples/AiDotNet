namespace AiDotNet.Helpers;

public static class VectorHelper
{
    public static Vector<T> CreateVector<T>(int size)
    {
        return new Vector<T>(size);
    }
}