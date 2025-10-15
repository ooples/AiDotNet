using System;

namespace AiDotNet.Helpers
{
    /// <summary>
    /// Provides compatibility helpers for array operations across different .NET versions
    /// </summary>
    internal static class ArrayHelper
    {
        /// <summary>
        /// Returns an empty array of the specified type.
        /// Provides compatibility for Array.Empty<T>() which is not available in .NET Framework 4.6.2
        /// </summary>
        /// <typeparam name="T">The type of the array elements</typeparam>
        /// <returns>An empty array of type T</returns>
        public static T[] Empty<T>()
        {
#if NET462
            return new T[0];
#else
            return Array.Empty<T>();
#endif
        }
    }
}