namespace AiDotNet.Extensions;

public static class EnumerableExtensions
{
    public static T RandomElement<T>(this IEnumerable<T> enumerable)
    {
        var list = enumerable as IList<T> ?? [.. enumerable];
        return list.Count == 0 ? MathHelper.GetNumericOperations<T>().Zero : list[new Random().Next(0, list.Count)];
    }
}