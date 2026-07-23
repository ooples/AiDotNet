#if !NET6_0_OR_GREATER
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.TrainingMonitoring;

/// <summary>
/// Polyfills for methods not available in older .NET Framework versions.
/// </summary>
internal static class FrameworkPolyfills
{
    /// <summary>
    /// Gets the relative path from a base path to a target path.
    /// Polyfill for Path.GetRelativePath which is not available in .NET Framework.
    /// </summary>
    public static string GetRelativePath(string relativeTo, string path)
    {
        if (string.IsNullOrEmpty(relativeTo))
            return path;

        if (string.IsNullOrEmpty(path))
            return string.Empty;

        // Normalize paths
        relativeTo = System.IO.Path.GetFullPath(relativeTo);
        path = System.IO.Path.GetFullPath(path);

        // Ensure trailing separator for directory
        if (!relativeTo.EndsWith(System.IO.Path.DirectorySeparatorChar.ToString()) &&
            !relativeTo.EndsWith(System.IO.Path.AltDirectorySeparatorChar.ToString()))
        {
            relativeTo += System.IO.Path.DirectorySeparatorChar;
        }

        Uri fromUri = new Uri(relativeTo);
        Uri toUri = new Uri(path);

        if (fromUri.Scheme != toUri.Scheme)
            return path;

        Uri relativeUri = fromUri.MakeRelativeUri(toUri);
        string relativePath = Uri.UnescapeDataString(relativeUri.ToString());

        if (toUri.Scheme.Equals("file", StringComparison.OrdinalIgnoreCase))
        {
            relativePath = relativePath.Replace('/', System.IO.Path.DirectorySeparatorChar);
        }

        return relativePath;
    }

    /// <summary>
    /// Returns a new enumerable that contains the last count elements from source.
    /// Polyfill for Enumerable.TakeLast which is not available in .NET Framework.
    /// </summary>
    public static IEnumerable<T> TakeLast<T>(this IEnumerable<T> source, int count)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));

        if (count <= 0)
        {
            yield break;
        }

        var list = source as IList<T>;
        if (list != null)
        {
            var start = Math.Max(0, list.Count - count);
            for (int i = start; i < list.Count; i++)
            {
                yield return list[i];
            }
            yield break;
        }

        // Fallback for non-list enumerables
        var queue = new Queue<T>(count);
        foreach (var item in source)
        {
            if (queue.Count >= count)
                queue.Dequeue();
            queue.Enqueue(item);
        }

        while (queue.Count > 0)
            yield return queue.Dequeue();
    }

    /// <summary>
    /// Returns the element with the minimum value according to the selector.
    /// Polyfill for Enumerable.MinBy which is not available before .NET 6.
    /// </summary>
    public static TSource MinBy<TSource, TKey>(this IEnumerable<TSource> source, Func<TSource, TKey> keySelector)
        where TKey : IComparable<TKey>
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (keySelector == null)
            throw new ArgumentNullException(nameof(keySelector));

        using (var enumerator = source.GetEnumerator())
        {
            if (!enumerator.MoveNext())
                throw new InvalidOperationException("Sequence contains no elements");

            var minElement = enumerator.Current;
            var minKey = keySelector(minElement);

            while (enumerator.MoveNext())
            {
                var currentKey = keySelector(enumerator.Current);
                if (currentKey.CompareTo(minKey) < 0)
                {
                    minKey = currentKey;
                    minElement = enumerator.Current;
                }
            }

            return minElement;
        }
    }

    /// <summary>
    /// Returns the element with the maximum value according to the selector.
    /// Polyfill for Enumerable.MaxBy which is not available before .NET 6.
    /// </summary>
    public static TSource MaxBy<TSource, TKey>(this IEnumerable<TSource> source, Func<TSource, TKey> keySelector)
        where TKey : IComparable<TKey>
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (keySelector == null)
            throw new ArgumentNullException(nameof(keySelector));

        using (var enumerator = source.GetEnumerator())
        {
            if (!enumerator.MoveNext())
                throw new InvalidOperationException("Sequence contains no elements");

            var maxElement = enumerator.Current;
            var maxKey = keySelector(maxElement);

            while (enumerator.MoveNext())
            {
                var currentKey = keySelector(enumerator.Current);
                if (currentKey.CompareTo(maxKey) > 0)
                {
                    maxKey = currentKey;
                    maxElement = enumerator.Current;
                }
            }

            return maxElement;
        }
    }

    /// <summary>
    /// Splits a string using a single character separator with options.
    /// Polyfill for string.Split(char, int, StringSplitOptions) which has different signature in .NET Framework.
    /// </summary>
    public static string[] SplitWithOptions(this string str, char separator, int count, StringSplitOptions options)
    {
        if (str == null)
            throw new ArgumentNullException(nameof(str));

        var parts = str.Split(new[] { separator }, count, options);
        return parts;
    }
}
#endif
