namespace AiDotNet.Serialization;

/// <summary>
/// Custom JSON converter for the Tensor&lt;T&gt; class to enable proper serialization and deserialization.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
/// <remarks>
/// <para>
/// This converter handles the serialization and deserialization of Tensor&lt;T&gt; objects using Newtonsoft.Json.
/// It leverages the existing SerializationHelper to maintain consistency with the binary serialization format.
/// </para>
/// <para><b>For Beginners:</b> This converter teaches the JSON serializer how to save and load our special Tensor type.
/// 
/// Tensors are multidimensional arrays used in machine learning for storing complex data like:
/// - Images (3D tensors with dimensions for height, width, and color channels)
/// - Video (4D tensors with an additional time dimension)
/// - Language data (embedding matrices)
/// 
/// This converter makes it possible to save and load these complex structures in a JSON format,
/// while maintaining consistency with our binary serialization format.
/// </para>
/// </remarks>
public class TensorJsonConverter<T> : JsonConverter
{
    /// <summary>
    /// Determines whether this converter can convert the specified object type.
    /// </summary>
    /// <param name="objectType">The type of the object to check.</param>
    /// <returns>True if this converter can convert the specified type; otherwise, false.</returns>
    public override bool CanConvert(Type objectType)
    {
        return objectType == typeof(Tensor<T>);
    }

    /// <summary>
    /// Writes the JSON representation of the Tensor&lt;T&gt; object.
    /// </summary>
    /// <param name="writer">The JsonWriter to write to.</param>
    /// <param name="value">The Tensor&lt;T&gt; to convert to JSON.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        if (value is not Tensor<T> tensor)
        {
            writer.WriteNull();
            return;
        }

        // Convert the tensor to a byte array using our existing serialization helper
        byte[] serializedData;
        using (var ms = new MemoryStream())
        using (var bw = new BinaryWriter(ms))
        {
            SerializationHelper<T>.SerializeTensor(bw, tensor);
            serializedData = ms.ToArray();
        }

        // Create a JSON object with tensor shape and Base64-encoded data
        JObject obj = new JObject();

        // Store the shape
        var shapeArray = new JArray();
        foreach (int dim in tensor.Shape)
        {
            shapeArray.Add(dim);
        }
        obj["Shape"] = shapeArray;

        // Store the serialized data
        obj["Data"] = Convert.ToBase64String(serializedData);

        obj.WriteTo(writer);
    }

    /// <summary>
    /// Reads the JSON representation of the Tensor&lt;T&gt; object.
    /// </summary>
    /// <param name="reader">The JsonReader to read from.</param>
    /// <param name="objectType">The type of the object to convert.</param>
    /// <param name="existingValue">The existing value of the object being read.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    /// <returns>The Tensor&lt;T&gt; object deserialized from JSON.</returns>
    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null)
            return null;

        JObject obj = JObject.Load(reader);

        // Read shape
        JArray? shapeArray = obj["Shape"] as JArray;
        if (shapeArray == null)
            return null;

        int[] shape = new int[shapeArray.Count];
        for (int i = 0; i < shapeArray.Count; i++)
        {
            shape[i] = shapeArray[i].Value<int>();
        }

        // If we just have shape information, create an empty tensor with that shape
        string? base64Data = obj["Data"]?.Value<string>();
        if (string.IsNullOrEmpty(base64Data))
            return new Tensor<T>(shape);

        // Decode and deserialize using our existing helper
        byte[] binaryData = Convert.FromBase64String(base64Data);

        using (var ms = new MemoryStream(binaryData))
        using (var br = new BinaryReader(ms))
        {
            // Skip the shape information since we already read it from JSON
            int rank = br.ReadInt32(); // Skip rank
            for (int i = 0; i < rank; i++)
            {
                br.ReadInt32(); // Skip dimension size
            }

            // Create a tensor with the shape we read from JSON
            var tensor = new Tensor<T>(shape);

            // Read all the values
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor[i] = SerializationHelper<T>.ReadValue(br);
            }

            return tensor;
        }
    }
}