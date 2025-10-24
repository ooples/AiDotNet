using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using AiDotNet.LinearAlgebra;
using System;

namespace AiDotNet.Serialization
{
    /// <summary>
    /// JSON converter for Tensor&lt;T&gt; types.
    /// Handles serialization and deserialization of tensor objects to/from JSON.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This class knows how to convert a Tensor (a multi-dimensional array
    /// of numbers) into JSON text format and back. It saves the shape (dimensions) and all the data values,
    /// so the tensor can be perfectly reconstructed later.</para>
    /// </remarks>
    public class TensorJsonConverter : JsonConverter
    {
        /// <summary>
        /// Determines whether this converter can handle the specified type.
        /// </summary>
        /// <param name="objectType">The type to check.</param>
        /// <returns>True if the type is Tensor&lt;T&gt;, false otherwise.</returns>
        public override bool CanConvert(Type objectType)
        {
            return objectType.IsGenericType &&
                   objectType.GetGenericTypeDefinition() == typeof(Tensor<>);
        }

        /// <summary>
        /// Writes a Tensor&lt;T&gt; object to JSON.
        /// </summary>
        /// <param name="writer">The JSON writer.</param>
        /// <param name="value">The tensor to serialize.</param>
        /// <param name="serializer">The JSON serializer.</param>
        /// <remarks>
        /// <para><b>For Beginners:</b> This method converts a Tensor into JSON format by saving:
        /// 1. The shape (dimensions of the tensor)
        /// 2. All the data in the tensor
        /// This allows the tensor to be saved to a file.</para>
        /// </remarks>
        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value == null)
            {
                writer.WriteNull();
                return;
            }

            var tensorType = value.GetType();
            var shapeProperty = tensorType.GetProperty("Shape");
            var lengthProperty = tensorType.GetProperty("Length");

            if (shapeProperty == null || lengthProperty == null)
            {
                throw new JsonSerializationException($"Cannot serialize tensor type {tensorType.Name}: missing required properties.");
            }

            var shape = (int[])shapeProperty.GetValue(value);
            var length = (int)lengthProperty.GetValue(value);

            // Get the ToArray method to extract all data
            var toArrayMethod = tensorType.GetMethod("ToArray");
            if (toArrayMethod == null)
            {
                throw new JsonSerializationException($"Cannot serialize tensor type {tensorType.Name}: missing ToArray method.");
            }

            var dataArray = toArrayMethod.Invoke(value, null);

            writer.WriteStartObject();
            writer.WritePropertyName("shape");
            serializer.Serialize(writer, shape);
            writer.WritePropertyName("data");
            serializer.Serialize(writer, dataArray);
            writer.WriteEndObject();
        }

        /// <summary>
        /// Reads a Tensor&lt;T&gt; object from JSON.
        /// </summary>
        /// <param name="reader">The JSON reader.</param>
        /// <param name="objectType">The type of object to create.</param>
        /// <param name="existingValue">The existing value (not used).</param>
        /// <param name="serializer">The JSON serializer.</param>
        /// <returns>A reconstructed Tensor&lt;T&gt; object.</returns>
        /// <remarks>
        /// <para><b>For Beginners:</b> This method reads JSON data and reconstructs a Tensor object.
        /// It reads the shape and data that were saved, then creates a new tensor with those exact values.</para>
        /// </remarks>
        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            if (reader.TokenType == JsonToken.Null)
            {
                return null;
            }

            var jObject = JObject.Load(reader);
            var shape = jObject["shape"].ToObject<int[]>();
            var dataToken = jObject["data"];

            // Get the element type (T) from Tensor<T>
            var elementType = objectType.GetGenericArguments()[0];

            // Convert data to array of the correct type
            var arrayType = elementType.MakeArrayType();
            var dataArray = dataToken.ToObject(arrayType);

            // Create tensor constructor: Tensor<T>(T[] data, int[] shape)
            // First, try the constructor with IEnumerable<T> and params int[]
            var enumerableType = typeof(System.Collections.Generic.IEnumerable<>).MakeGenericType(elementType);
            var tensorConstructor = objectType.GetConstructor(new[] { enumerableType, typeof(int[]) });

            if (tensorConstructor == null)
            {
                throw new JsonSerializationException($"Cannot find constructor for {objectType.Name}(IEnumerable<{elementType.Name}>, int[])");
            }

            var tensor = tensorConstructor.Invoke(new object[] { dataArray, shape });

            return tensor;
        }
    }
}
