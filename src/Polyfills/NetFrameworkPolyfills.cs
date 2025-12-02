// Polyfills for .NET Framework 4.7.1 to support modern C# features

#if !NET5_0_OR_GREATER

using System.Collections.Generic;
using System.Runtime.CompilerServices;

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

namespace System
{
    /// <summary>
    /// Polyfills for Math methods missing from .NET Framework.
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
            return Math.Log(value) / Math.Log(2.0);
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
            if (value < min) return min;
            if (value > max) return max;
            return value;
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
            if (value < min) return min;
            if (value > max) return max;
            return value;
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
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }
    }

    /// <summary>
    /// Extension methods for Array to add missing methods from newer .NET versions.
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
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = value;
            }
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
            for (int i = startIndex; i < startIndex + count && i < array.Length; i++)
            {
                array[i] = value;
            }
        }
    }
}

#endif
