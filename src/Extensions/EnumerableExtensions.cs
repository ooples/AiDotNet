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
        return list.Count == 0 ? MathHelper.GetNumericOperations<T>().Zero : list[new Random().Next(0, list.Count)];
    }

    /// <summary>
    /// Deconstructs a KeyValuePair into its key and value components.
    /// </summary>
    /// <typeparam name="TKey">The type of the key.</typeparam>
    /// <typeparam name="TValue">The type of the value.</typeparam>
    /// <param name="kvp">The KeyValuePair to deconstruct.</param>
    /// <param name="key">The key component.</param>
    /// <param name="value">The value component.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This extension method allows you to use tuple deconstruction syntax
    /// with KeyValuePair objects in .NET Framework 4.6.2, which doesn't natively support this feature.
    ///
    /// Example usage:
    /// <code>
    /// var dict = new Dictionary&lt;string, int&gt; { {"apple", 5}, {"banana", 3} };
    /// foreach (var (key, value) in dict)
    /// {
    ///     Console.WriteLine($"{key}: {value}");
    /// }
    /// </code>
    /// </remarks>
    public static void Deconstruct<TKey, TValue>(this KeyValuePair<TKey, TValue> kvp, out TKey key, out TValue value)
    {
        key = kvp.Key;
        value = kvp.Value;
    }

    // ====================================================================
    // .NET Framework 4.6.2 Compatibility Methods
    // ====================================================================
    //
    // The following methods are only included when targeting .NET Framework 4.6.2
    // They are built into System.Linq.Enumerable in .NET Core 2.0+ and .NET 6.0+
    //
    // Conditional compilation prevents CS0121 ambiguous call errors when
    // targeting modern .NET versions while maintaining compatibility with
    // .NET Framework 4.6.2
    //
    // ====================================================================

#if NET462
    /// <summary>
    /// Returns the last N elements from a sequence.
    /// </summary>
    /// <typeparam name="T">The type of elements in the sequence.</typeparam>
    /// <param name="source">The sequence to return elements from.</param>
    /// <param name="count">The number of elements to return from the end of the sequence.</param>
    /// <returns>An IEnumerable containing the last N elements.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method gets the last few items from a collection, similar to
    /// looking at the last entries in a list. It's useful when you only care about recent data.
    ///
    /// <b>Note:</b> This method is only compiled for .NET Framework 4.6.2. In .NET 6.0+ and .NET Core 2.0+,
    /// use the built-in System.Linq.Enumerable.TakeLast() method instead.
    ///
    /// Example usage:
    /// <code>
    /// var numbers = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    /// var lastThree = numbers.TakeLast(3); // Returns { 8, 9, 10 }
    /// </code>
    /// </remarks>
    public static IEnumerable<T> TakeLast<T>(this IEnumerable<T> source, int count)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative.");

        var list = source as IList<T> ?? source.ToList();
        var startIndex = Math.Max(0, list.Count - count);

        for (int i = startIndex; i < list.Count; i++)
        {
            yield return list[i];
        }
    }

    /// <summary>
    /// Creates a HashSet from an IEnumerable.
    /// </summary>
    /// <typeparam name="T">The type of elements in the sequence.</typeparam>
    /// <param name="source">The sequence to create a HashSet from.</param>
    /// <returns>A HashSet containing all unique elements from the source sequence.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A HashSet is a collection that only contains unique values (no duplicates).
    /// This method converts any collection into a HashSet, automatically removing duplicates.
    ///
    /// <b>Note:</b> This method is only compiled for .NET Framework 4.6.2. In .NET 6.0+ and .NET Core 2.0+,
    /// use the built-in System.Linq.Enumerable.ToHashSet() method instead.
    ///
    /// Example usage:
    /// <code>
    /// var numbers = new[] { 1, 2, 2, 3, 3, 3, 4 };
    /// var uniqueNumbers = numbers.ToHashSet(); // Returns { 1, 2, 3, 4 }
    /// </code>
    /// </remarks>
    public static HashSet<T> ToHashSet<T>(this IEnumerable<T> source)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));

        return new HashSet<T>(source);
    }
#endif

    /// <summary>
    /// Gets the value associated with the specified key, or a default value if the key doesn't exist.
    /// </summary>
    /// <typeparam name="TKey">The type of keys in the dictionary.</typeparam>
    /// <typeparam name="TValue">The type of values in the dictionary.</typeparam>
    /// <param name="dictionary">The dictionary to search.</param>
    /// <param name="key">The key to look up.</param>
    /// <param name="defaultValue">The default value to return if the key is not found.</param>
    /// <returns>The value associated with the key, or the default value if not found.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method safely gets a value from a dictionary without throwing
    /// an error if the key doesn't exist. Instead, it returns a default value you specify.
    ///
    /// Example usage:
    /// <code>
    /// var scores = new Dictionary&lt;string, int&gt; { {"Alice", 95}, {"Bob", 87} };
    /// var aliceScore = scores.GetValueOrDefault("Alice", 0); // Returns 95
    /// var charlieScore = scores.GetValueOrDefault("Charlie", 0); // Returns 0 (default)
    /// </code>
    /// </remarks>
    public static TValue? GetValueOrDefault<TKey, TValue>(this Dictionary<TKey, TValue> dictionary, TKey key, TValue? defaultValue = default)
        where TKey : notnull
    {
        if (dictionary == null)
            throw new ArgumentNullException(nameof(dictionary));

        return dictionary.TryGetValue(key, out var value) ? value : defaultValue;
    }
}