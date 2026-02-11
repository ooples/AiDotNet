using System;
using AiDotNet.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
        /// <returns>True if the type is Tensor&lt;T&gt; or a subclass thereof, false otherwise.</returns>
        /// <remarks>
        /// This method walks the inheritance chain to support subclasses of Tensor&lt;T&gt;.
        /// </remarks>
        public override bool CanConvert(Type objectType)
        {
            // Walk the inheritance chain to support subclasses
            Type? currentType = objectType;
            while (currentType != null)
            {
                if (currentType.IsGenericType &&
                    currentType.GetGenericTypeDefinition() == typeof(Tensor<>))
                {
                    return true;
                }
                currentType = currentType.BaseType;
            }
            return false;
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

            object? shapeObj = shapeProperty.GetValue(value);
            object? lengthObj = lengthProperty.GetValue(value);
            if (shapeObj == null || lengthObj == null)
            {
                throw new JsonSerializationException($"Cannot serialize tensor: Shape or Length property returned null.");
            }
            var shape = (int[])shapeObj;
            var length = (int)lengthObj;

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
            var shape = jObject["shape"]?.ToObject<int[]>();
            var dataToken = jObject["data"];

            if (shape == null)
            {
                throw new JsonSerializationException("Tensor JSON must contain 'shape' property.");
            }

            // Handle scalar tensors: empty shape is treated as a 1-element tensor [1]
            // This provides backward compatibility with legacy JSON that may have empty shapes
            if (shape.Length == 0)
            {
                shape = new[] { 1 };
            }

            // Validate that all dimensions are non-negative
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] < 0)
                {
                    throw new JsonSerializationException(
                        $"Tensor 'shape' dimension at index {i} must be non-negative, but was {shape[i]}.");
                }
            }

            // Get the element type (T) from Tensor<T>
            var elementType = objectType.GetGenericArguments()[0];

            // Convert data to array of the correct type
            var arrayType = elementType.MakeArrayType();
            var dataArray = (Array?)dataToken?.ToObject(arrayType);

            if (dataArray == null)
            {
                throw new JsonSerializationException("Tensor JSON must contain 'data' property.");
            }

            // Validate that flattened data length matches product of shape dimensions
            int expectedLength = 1;
            foreach (int dim in shape)
            {
                expectedLength *= dim;
            }

            if (dataArray.Length != expectedLength)
            {
                throw new JsonSerializationException(
                    $"Tensor data length mismatch: expected {expectedLength} elements (from shape [{string.Join(", ", shape)}]), " +
                    $"but got {dataArray.Length} elements.");
            }

            // Prefer the current tensor API: Tensor(int[] dimensions, Vector<T> data).
            var vectorType = typeof(AiDotNet.Tensors.LinearAlgebra.Vector<>).MakeGenericType(elementType);
            var tensorConstructor = objectType.GetConstructor(new[] { typeof(int[]), vectorType });

            if (tensorConstructor != null)
            {
                var vector = Activator.CreateInstance(vectorType, new object[] { dataArray })
                    ?? throw new JsonSerializationException($"Cannot create Vector<{elementType.Name}> instance for tensor deserialization.");

                return tensorConstructor.Invoke(new object[] { shape, vector });
            }

            // Fallback: Tensor(int[] dimensions) + CopyFromArray.
            tensorConstructor = objectType.GetConstructor(new[] { typeof(int[]) });
            if (tensorConstructor != null)
            {
                var tensor = tensorConstructor.Invoke(new object[] { shape })
                    ?? throw new JsonSerializationException($"Constructor for {objectType.Name} returned null.");

                // Use the public CopyFromArray method to populate the tensor data
                var copyFromArrayMethod = objectType.GetMethod("CopyFromArray", new[] { dataArray.GetType() });
                if (copyFromArrayMethod == null)
                {
                    throw new JsonSerializationException($"Cannot deserialize tensor type {objectType.Name}: missing CopyFromArray method.");
                }

                copyFromArrayMethod.Invoke(tensor, new object[] { dataArray });
                return tensor;
            }

            throw new JsonSerializationException(
                $"Cannot find suitable constructor for {objectType.Name}. " +
                $"Expected constructor with signature (int[], Vector<{elementType.Name}>) or (int[]).");
        }
    }
}
