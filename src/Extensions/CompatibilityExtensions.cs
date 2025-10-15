using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Extensions
{
    /// <summary>
    /// Extension methods for .NET Framework compatibility
    /// </summary>
    public static class CompatibilityExtensions
    {
        /// <summary>
        /// Returns a specified number of contiguous elements from the end of a sequence.
        /// </summary>
        /// <typeparam name="TSource">The type of the elements in the source sequence.</typeparam>
        /// <param name="source">The sequence to return elements from.</param>
        /// <param name="count">The number of elements to return.</param>
        /// <returns>An IEnumerable{TSource} that contains the specified number of elements from the end of the input sequence.</returns>
        public static IEnumerable<TSource> TakeLast<TSource>(this IEnumerable<TSource> source, int count)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            
            if (count <= 0)
                return Enumerable.Empty<TSource>();
            
            var list = source as IList<TSource> ?? source.ToList();
            
            if (count >= list.Count)
                return list;
            
            return list.Skip(list.Count - count);
        }
        
        /// <summary>
        /// Returns a new enumerable collection that contains the elements from source with the last count elements of the source collection omitted.
        /// </summary>
        /// <typeparam name="TSource">The type of the elements in the source sequence.</typeparam>
        /// <param name="source">The sequence to return elements from.</param>
        /// <param name="count">The number of elements to omit from the end of the sequence.</param>
        /// <returns>An IEnumerable{TSource} that contains the elements from source minus the last count elements.</returns>
        public static IEnumerable<TSource> SkipLast<TSource>(this IEnumerable<TSource> source, int count)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            
            if (count <= 0)
                return source;
            
            var list = source as IList<TSource> ?? source.ToList();
            
            if (count >= list.Count)
                return Enumerable.Empty<TSource>();
            
            return list.Take(list.Count - count);
        }
    }
}