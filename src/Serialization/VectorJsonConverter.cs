using System;
using AiDotNet.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serialization
{
    /// <summary>
    /// JSON converter for Vector&lt;T&gt; types.
    /// Handles serialization and deserialization of vector objects to/from JSON.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This class knows how to convert a Vector (a list of numbers) into
    /// JSON text format and back. It saves the length and all the data values, so the vector can be
    /// perfectly reconstructed later.</para>
    /// </remarks>
    public class VectorJsonConverter : JsonConverter
    {
        /// <summary>
        /// Determines whether this converter can handle the specified type.
        /// </summary>
        /// <param name="objectType">The type to check.</param>
        /// <returns>True if the type is Vector&lt;T&gt; or a subclass thereof, false otherwise.</returns>
        /// <remarks>
        /// This method walks the inheritance chain to support subclasses of Vector&lt;T&gt;.
        /// </remarks>
        public override bool CanConvert(Type objectType)
        {
            // Walk the inheritance chain to support subclasses
            Type? currentType = objectType;
            while (currentType != null)
            {
                if (currentType.IsGenericType &&
                    currentType.GetGenericTypeDefinition() == typeof(Vector<>))
                {
                    return true;
                }
                currentType = currentType.BaseType;
            }
            return false;
        }

        /// <summary>
        /// Writes a Vector&lt;T&gt; object to JSON.
        /// </summary>
        /// <param name="writer">The JSON writer.</param>
        /// <param name="value">The vector to serialize.</param>
        /// <param name="serializer">The JSON serializer.</param>
        /// <remarks>
        /// <para><b>For Beginners:</b> This method converts a Vector into JSON format by saving:
        /// 1. The length (number of elements)
        /// 2. All the data in the vector
        /// This allows the vector to be saved to a file.</para>
        /// </remarks>
        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value == null)
            {
                writer.WriteNull();
                return;
            }

            var vectorType = value.GetType();
            var lengthProperty = vectorType.GetProperty("Length");
            var indexer = vectorType.GetProperty("Item", new[] { typeof(int) });

            if (lengthProperty == null || indexer == null)
            {
                throw new JsonSerializationException($"Cannot serialize vector type {vectorType.Name}: missing required properties.");
            }

            object? lengthObj = lengthProperty.GetValue(value);
            if (lengthObj == null)
            {
                throw new JsonSerializationException($"Cannot serialize vector: Length property returned null.");
            }
            var length = (int)lengthObj;

            writer.WriteStartObject();
            writer.WritePropertyName("length");
            writer.WriteValue(length);
            writer.WritePropertyName("data");
            writer.WriteStartArray();

            for (int i = 0; i < length; i++)
            {
                var cellValue = indexer.GetValue(value, new object[] { i });
                serializer.Serialize(writer, cellValue);
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        /// <summary>
        /// Reads a Vector&lt;T&gt; object from JSON.
        /// </summary>
        /// <param name="reader">The JSON reader.</param>
        /// <param name="objectType">The type of object to create.</param>
        /// <param name="existingValue">The existing value (not used).</param>
        /// <param name="serializer">The JSON serializer.</param>
        /// <returns>A reconstructed Vector&lt;T&gt; object.</returns>
        /// <remarks>
        /// <para><b>For Beginners:</b> This method reads JSON data and reconstructs a Vector object.
        /// It reads the length and data that were saved, then creates a new vector with those exact values.</para>
        /// </remarks>
        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            if (reader.TokenType == JsonToken.Null)
            {
                return null;
            }

            var jObject = JObject.Load(reader);

            // Validate that 'length' token exists and is an integer
            var lengthToken = jObject["length"];
            if (lengthToken == null || lengthToken.Type != JTokenType.Integer)
            {
                throw new JsonSerializationException("Vector JSON must contain 'length' property as an integer.");
            }
            int length = lengthToken.Value<int>();

            // Validate that 'data' token exists and is a JArray
            var dataToken = jObject["data"];
            if (dataToken == null || dataToken.Type != JTokenType.Array)
            {
                throw new JsonSerializationException("Vector JSON must contain 'data' property as an array.");
            }

            var dataArray = (JArray)dataToken;

            // Validate that data count matches length
            if (dataArray.Count != length)
            {
                throw new JsonSerializationException(
                    $"Vector data count mismatch: length property is {length}, but data array contains {dataArray.Count} elements.");
            }

            // Get the element type (T) from Vector<T>
            var elementType = objectType.GetGenericArguments()[0];
            var arrayType = elementType.MakeArrayType();

            // Build a T[] array using serializer.Deserialize to respect converters
            var array = Array.CreateInstance(elementType, length);
            for (int i = 0; i < length; i++)
            {
                using (var tokenReader = dataArray[i].CreateReader())
                {
                    var element = serializer.Deserialize(tokenReader, elementType);
                    array.SetValue(element, i);
                }
            }

            // Try to find a writable indexer first (for mutable vectors)
            var indexer = objectType.GetProperty("Item", new[] { typeof(int) });
            bool hasWritableIndexer = indexer != null && indexer.CanWrite;

            if (hasWritableIndexer)
            {
                // Use constructor with int length and populate via indexer
                var vectorConstructor = objectType.GetConstructor(new[] { typeof(int) });
                if (vectorConstructor != null)
                {
                    var vector = vectorConstructor.Invoke(new object[] { length });

                    for (int i = 0; i < length; i++)
                    {
                        indexer!.SetValue(vector, array.GetValue(i), new object[] { i });
                    }

                    return vector;
                }
            }

            // Fallback: Try factory method Vector<T>.FromArray(T[])
            var fromArrayMethod = objectType.GetMethod("FromArray",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static,
                null,
                new[] { arrayType },
                null);

            if (fromArrayMethod != null)
            {
                return fromArrayMethod.Invoke(null, new object[] { array });
            }

            // Fallback: Try constructor taking T[]
            var arrayConstructor = objectType.GetConstructor(new[] { arrayType });
            if (arrayConstructor != null)
            {
                return arrayConstructor.Invoke(new object[] { array });
            }

            // No suitable construction method found
            throw new JsonSerializationException(
                $"Cannot construct {objectType.Name} from JSON. " +
                $"Type requires either: a constructor taking int with writable indexer, " +
                $"a static FromArray({elementType.Name}[]) method, or a constructor taking {elementType.Name}[].");
        }
    }
}
