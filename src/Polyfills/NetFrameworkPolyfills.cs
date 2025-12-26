// Polyfills for .NET Framework 4.7.1 to support modern C# features

using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;

#if !NET5_0_OR_GREATER

namespace System.Collections.Generic
{
    /// <summary>
    /// Extension methods for KeyValuePair to support deconstruction in .NET Framework.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In modern C#, you can write:
    /// <code>foreach (var (key, value) in dictionary) { ... }</code>
    /// This polyfill enables that syntax in .NET Framework.
    /// </para>
    /// </remarks>
    public static class KeyValuePairExtensions
    {
        /// <summary>
        /// Deconstructs a KeyValuePair into its key and value components.
        /// </summary>
        /// <typeparam name="TKey">The type of the key.</typeparam>
        /// <typeparam name="TValue">The type of the value.</typeparam>
        /// <param name="kvp">The KeyValuePair to deconstruct.</param>
        /// <param name="key">The key component.</param>
        /// <param name="value">The value component.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<TKey, TValue>(
            this KeyValuePair<TKey, TValue> kvp,
            out TKey key,
            out TValue value)
        {
            key = kvp.Key;
            value = kvp.Value;
        }
    }

    /// <summary>
    /// Extension methods for Dictionary to add missing methods from newer .NET versions.
    /// </summary>
    public static class DictionaryExtensions
    {
        /// <summary>
        /// Gets the value associated with the specified key, or the default value if not found.
        /// </summary>
        /// <typeparam name="TKey">The type of keys in the dictionary.</typeparam>
        /// <typeparam name="TValue">The type of values in the dictionary.</typeparam>
        /// <param name="dictionary">The dictionary to search.</param>
        /// <param name="key">The key to look up.</param>
        /// <returns>The value if found; otherwise, the default value for TValue.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TValue? GetValueOrDefault<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary,
            TKey key) where TKey : notnull
        {
            return dictionary.TryGetValue(key, out var value) ? value : default;
        }

        /// <summary>
        /// Gets the value associated with the specified key, or the specified default value if not found.
        /// </summary>
        /// <typeparam name="TKey">The type of keys in the dictionary.</typeparam>
        /// <typeparam name="TValue">The type of values in the dictionary.</typeparam>
        /// <param name="dictionary">The dictionary to search.</param>
        /// <param name="key">The key to look up.</param>
        /// <param name="defaultValue">The default value to return if the key is not found.</param>
        /// <returns>The value if found; otherwise, the specified default value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TValue GetValueOrDefault<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary,
            TKey key,
            TValue defaultValue) where TKey : notnull
        {
            return dictionary.TryGetValue(key, out var value) ? value : defaultValue;
        }

        /// <summary>
        /// Gets the value associated with the specified key, or the default value if not found.
        /// </summary>
        /// <typeparam name="TKey">The type of keys in the dictionary.</typeparam>
        /// <typeparam name="TValue">The type of values in the dictionary.</typeparam>
        /// <param name="dictionary">The dictionary to search.</param>
        /// <param name="key">The key to look up.</param>
        /// <returns>The value if found; otherwise, the default value for TValue.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TValue? GetValueOrDefault<TKey, TValue>(
            this IDictionary<TKey, TValue> dictionary,
            TKey key) where TKey : notnull
        {
            return dictionary.TryGetValue(key, out var value) ? value : default;
        }

        /// <summary>
        /// Gets the value associated with the specified key, or the specified default value if not found.
        /// </summary>
        /// <typeparam name="TKey">The type of keys in the dictionary.</typeparam>
        /// <typeparam name="TValue">The type of values in the dictionary.</typeparam>
        /// <param name="dictionary">The dictionary to search.</param>
        /// <param name="key">The key to look up.</param>
        /// <param name="defaultValue">The default value to return if the key is not found.</param>
        /// <returns>The value if found; otherwise, the specified default value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TValue GetValueOrDefault<TKey, TValue>(
            this IDictionary<TKey, TValue> dictionary,
            TKey key,
            TValue defaultValue) where TKey : notnull
        {
            return dictionary.TryGetValue(key, out var value) ? value : defaultValue;
        }
    }

    /// <summary>
    /// Extension methods for IEnumerable to add ToHashSet functionality.
    /// </summary>
    public static class EnumerableExtensions
    {
        /// <summary>
        /// Creates a HashSet from an IEnumerable.
        /// </summary>
        /// <typeparam name="T">The type of elements.</typeparam>
        /// <param name="source">The source sequence.</param>
        /// <returns>A new HashSet containing the elements from the source.</returns>
        public static HashSet<T> ToHashSet<T>(this IEnumerable<T> source)
        {
            return new HashSet<T>(source);
        }

        /// <summary>
        /// Creates a HashSet from an IEnumerable using the specified comparer.
        /// </summary>
        /// <typeparam name="T">The type of elements.</typeparam>
        /// <param name="source">The source sequence.</param>
        /// <param name="comparer">The equality comparer to use.</param>
        /// <returns>A new HashSet containing the elements from the source.</returns>
        public static HashSet<T> ToHashSet<T>(this IEnumerable<T> source, IEqualityComparer<T>? comparer)
        {
            return new HashSet<T>(source, comparer);
        }
    }
}

#endif

// The following polyfills are available for ALL frameworks
// They provide a consistent API that works the same way on both .NET Framework and modern .NET

namespace System
{
    /// <summary>
    /// Polyfills for Math methods missing from .NET Framework.
    /// Available for all frameworks - delegates to standard library on modern .NET.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some Math methods like Clamp and Log2 don't exist in .NET Framework.
    /// This class provides these methods so code can work across all .NET versions.
    /// </para>
    /// </remarks>
    public static class MathPolyfill
    {
        /// <summary>
        /// Computes the base-2 logarithm of a number.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The base-2 logarithm.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Log2(double value)
        {
            return MathHelper.Log2(value);
        }

        /// <summary>
        /// Clamps a value between a minimum and maximum value.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <returns>The clamped value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Clamp(int value, int min, int max)
        {
            return MathHelper.Clamp(value, min, max);
        }

        /// <summary>
        /// Clamps a value between a minimum and maximum value.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <returns>The clamped value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp(float value, float min, float max)
        {
            return MathHelper.Clamp(value, min, max);
        }

        /// <summary>
        /// Clamps a value between a minimum and maximum value.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <returns>The clamped value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Clamp(double value, double min, double max)
        {
            return MathHelper.Clamp(value, min, max);
        }
    }

    /// <summary>
    /// Extension methods for Array to add missing methods from newer .NET versions.
    /// Available for all frameworks.
    /// </summary>
    public static class ArrayPolyfill
    {
        /// <summary>
        /// Fills an array with a specified value.
        /// </summary>
        /// <typeparam name="T">The type of array elements.</typeparam>
        /// <param name="array">The array to fill.</param>
        /// <param name="value">The value to fill with.</param>
        public static void Fill<T>(T[] array, T value)
        {
#if NET5_0_OR_GREATER
            Array.Fill(array, value);
#else
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = value;
            }
#endif
        }

        /// <summary>
        /// Fills a portion of an array with a specified value.
        /// </summary>
        /// <typeparam name="T">The type of array elements.</typeparam>
        /// <param name="array">The array to fill.</param>
        /// <param name="value">The value to fill with.</param>
        /// <param name="startIndex">The starting index.</param>
        /// <param name="count">The number of elements to fill.</param>
        public static void Fill<T>(T[] array, T value, int startIndex, int count)
        {
#if NET5_0_OR_GREATER
            Array.Fill(array, value, startIndex, count);
#else
            for (int i = startIndex; i < startIndex + count && i < array.Length; i++)
            {
                array[i] = value;
            }
#endif
        }
    }
}

namespace System.IO
{
    /// <summary>
    /// Polyfills for File methods missing from .NET Framework.
    /// Available for all frameworks - delegates to standard library on modern .NET.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some async File methods like ReadAllLinesAsync don't exist in .NET Framework.
    /// This class provides these methods so code can work across all .NET versions.
    /// </para>
    /// </remarks>
    public static class FilePolyfill
    {
        /// <summary>
        /// Asynchronously reads all lines from a file.
        /// </summary>
        /// <param name="path">The path to the file.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task that represents the asynchronous read operation, containing all lines.</returns>
        public static async Threading.Tasks.Task<string[]> ReadAllLinesAsync(
            string path,
            Threading.CancellationToken cancellationToken = default)
        {
#if NET5_0_OR_GREATER
            return await File.ReadAllLinesAsync(path, cancellationToken);
#else
            var lines = new System.Collections.Generic.List<string>();
            using (var reader = new StreamReader(path))
            {
                string? line;
                while ((line = await reader.ReadLineAsync()) is not null)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    lines.Add(line);
                }
            }
            return lines.ToArray();
#endif
        }

        /// <summary>
        /// Asynchronously reads all text from a file.
        /// </summary>
        /// <param name="path">The path to the file.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task that represents the asynchronous read operation, containing all text.</returns>
        public static async Threading.Tasks.Task<string> ReadAllTextAsync(
            string path,
            Threading.CancellationToken cancellationToken = default)
        {
#if NET5_0_OR_GREATER
            return await File.ReadAllTextAsync(path, cancellationToken);
#else
            using (var reader = new StreamReader(path))
            {
                cancellationToken.ThrowIfCancellationRequested();
                return await reader.ReadToEndAsync();
            }
#endif
        }

        /// <summary>
        /// Asynchronously writes text to a file, creating the file if it doesn't exist or overwriting it if it does.
        /// </summary>
        /// <param name="path">The path to the file.</param>
        /// <param name="contents">The text to write.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task that represents the asynchronous write operation.</returns>
        public static async Threading.Tasks.Task WriteAllTextAsync(
            string path,
            string contents,
            Threading.CancellationToken cancellationToken = default)
        {
#if NET5_0_OR_GREATER
            await File.WriteAllTextAsync(path, contents, cancellationToken);
#else
            cancellationToken.ThrowIfCancellationRequested();
            using (var writer = new StreamWriter(path, false))
            {
                await writer.WriteAsync(contents);
            }
#endif
        }

        /// <summary>
        /// Asynchronously writes lines to a file, creating the file if it doesn't exist or overwriting it if it does.
        /// </summary>
        /// <param name="path">The path to the file.</param>
        /// <param name="contents">The lines to write.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task that represents the asynchronous write operation.</returns>
        public static async Threading.Tasks.Task WriteAllLinesAsync(
            string path,
            System.Collections.Generic.IEnumerable<string> contents,
            Threading.CancellationToken cancellationToken = default)
        {
#if NET5_0_OR_GREATER
            await File.WriteAllLinesAsync(path, contents, cancellationToken);
#else
            cancellationToken.ThrowIfCancellationRequested();
            using (var writer = new StreamWriter(path, false))
            {
                foreach (var line in contents)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    await writer.WriteLineAsync(line);
                }
            }
#endif
        }
    }
}
