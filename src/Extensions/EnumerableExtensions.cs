namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for IEnumerable collections to enhance their functionality.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Extension methods are special methods that add new capabilities to existing types
/// without modifying the original code. This class adds useful operations to collections in your code.
/// </remarks>
public static class EnumerableExtensions
{
    /// <summary>
    /// Returns a random element from a collection.
    /// </summary>
    /// <typeparam name="T">The type of elements in the collection.</typeparam>
    /// <param name="enumerable">The collection to select a random element from.</param>
    /// <returns>
    /// A randomly selected element from the collection. If the collection is empty,
    /// returns the numeric zero value for type T.
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method picks one random item from your collection, similar to
    /// drawing a name from a hat. If your collection is empty, it returns zero (or the equivalent
    /// for the data type you're using).
    /// 
    /// The method works with any collection type (arrays, lists, etc.) and handles empty collections
    /// safely by returning a default value instead of causing an error.
    /// 
    /// Example usage:
    /// <code>
    /// var numbers = new[] { 1, 2, 3, 4, 5 };
    /// var randomNumber = numbers.RandomElement(); // Might return any number from the array
    /// </code>
    /// </remarks>
    public static T RandomElement<T>(this IEnumerable<T> enumerable)
    {
        var list = enumerable as IList<T> ?? [.. enumerable];
        return list.Count == 0 ? MathHelper.GetNumericOperations<T>().Zero : list[RandomHelper.CreateSecureRandom().Next(0, list.Count)];
    }
}
