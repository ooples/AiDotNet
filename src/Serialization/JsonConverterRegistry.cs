using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

namespace AiDotNet.Serialization
{
    /// <summary>
    /// Registry for JSON converters used in model serialization.
    /// Manages custom converters for complex types like Matrix, Vector, and Tensor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This class helps convert complex data structures (like matrices and tensors)
    /// into JSON format so they can be saved to files and loaded later. JSON is a text format that's easy to
    /// read and write, making it perfect for saving machine learning models.</para>
    /// </remarks>
    public static class JsonConverterRegistry
    {
        private static readonly List<JsonConverter> _converters = new List<JsonConverter>();
        private static readonly object _lock = new object();
        private static bool _initialized = false;

        /// <summary>
        /// Registers all default converters for common types.
        /// This method is thread-safe and can be called multiple times safely.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> This sets up the converters needed to save and load matrices,
        /// vectors, and tensors. Call this once before serializing your model.</para>
        /// </remarks>
        public static void RegisterAllConverters()
        {
            lock (_lock)
            {
                if (_initialized)
                {
                    return;
                }

                _converters.Clear();

                // Register converters for Matrix, Vector, and Tensor types
                _converters.Add(new MatrixJsonConverter());
                _converters.Add(new VectorJsonConverter());
                _converters.Add(new TensorJsonConverter());

                _initialized = true;
            }
        }

        /// <summary>
        /// Gets all registered JSON converters.
        /// </summary>
        /// <returns>A list of all registered converters.</returns>
        /// <remarks>
        /// <para><b>For Beginners:</b> This returns the list of all converters that have been registered.
        /// These converters tell the JSON serializer how to handle special types.</para>
        /// </remarks>
        public static List<JsonConverter> GetAllConverters()
        {
            lock (_lock)
            {
                if (!_initialized)
                {
                    RegisterAllConverters();
                }

                return new List<JsonConverter>(_converters);
            }
        }

        /// <summary>
        /// Gets converters that can handle the specified type.
        /// </summary>
        /// <typeparam name="T">The type to get converters for.</typeparam>
        /// <returns>A list of converters that can handle type T.</returns>
        /// <remarks>
        /// <para><b>For Beginners:</b> This finds the right converter for a specific data type.
        /// For example, if you're working with doubles, this will return converters that know
        /// how to handle matrices, vectors, and tensors of doubles.</para>
        /// </remarks>
        public static List<JsonConverter> GetConvertersForType<T>()
        {
            lock (_lock)
            {
                if (!_initialized)
                {
                    RegisterAllConverters();
                }

                // Return all converters - they will check CanConvert themselves
                return new List<JsonConverter>(_converters);
            }
        }

        /// <summary>
        /// Registers a custom JSON converter.
        /// </summary>
        /// <param name="converter">The converter to register.</param>
        /// <exception cref="ArgumentNullException">Thrown when converter is null.</exception>
        /// <remarks>
        /// <para><b>For Beginners:</b> This allows you to add your own custom converter if you need
        /// to serialize a type that isn't already supported.</para>
        /// </remarks>
        public static void RegisterConverter(JsonConverter converter)
        {
            if (converter == null)
            {
                throw new ArgumentNullException(nameof(converter));
            }

            lock (_lock)
            {
                if (!_converters.Contains(converter))
                {
                    _converters.Add(converter);
                }
            }
        }

        /// <summary>
        /// Clears all registered converters.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> This removes all converters and resets the registry.
        /// Useful for testing or if you need to start fresh.</para>
        /// </remarks>
        public static void ClearConverters()
        {
            lock (_lock)
            {
                _converters.Clear();
                _initialized = false;
            }
        }
    }
}
