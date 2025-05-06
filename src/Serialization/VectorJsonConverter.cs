namespace AiDotNet.Serialization;

/// <summary>
/// Custom JSON converter for the Vector&lt;T&gt; class to enable proper serialization and deserialization.
/// </summary>
/// <typeparam name="T">The numeric type of the vector elements.</typeparam>
/// <remarks>
/// <para>
/// This converter handles the serialization and deserialization of Vector&lt;T&gt; objects using Newtonsoft.Json.
/// It leverages the existing SerializationHelper to maintain consistency with the binary serialization format.
/// </para>
/// <para><b>For Beginners:</b> This converter tells the JSON serializer how to save and load our special Vector type.
/// 
/// When working with custom collection types like our Vector class:
/// - The default JSON serializer doesn't know how to create instances properly
/// - We need to provide custom instructions for saving and loading vectors
/// - This converter provides those instructions by using our existing serialization helpers
/// 
/// This ensures our JSON and binary serialization formats remain consistent.
/// </para>
/// </remarks>
public class VectorJsonConverter<T> : JsonConverter
{
    /// <summary>
    /// Determines whether this converter can convert the specified object type.
    /// </summary>
    /// <param name="objectType">The type of the object to check.</param>
    /// <returns>True if this converter can convert the specified type; otherwise, false.</returns>
    public override bool CanConvert(Type objectType)
    {
        return objectType == typeof(Vector<T>);
    }

    /// <summary>
    /// Writes the JSON representation of the Vector&lt;T&gt; object.
    /// </summary>
    /// <param name="writer">The JsonWriter to write to.</param>
    /// <param name="value">The Vector&lt;T&gt; to convert to JSON.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        if (value is not Vector<T> vector)
        {
            writer.WriteNull();
            return;
        }

        // Convert the vector to a byte array using our existing serialization helper
        byte[] serializedData = SerializationHelper<T>.SerializeVector(vector);

        // Create a JSON object with vector length and Base64-encoded data
        JObject obj = new JObject();
        obj["Length"] = vector.Length;
        obj["Data"] = Convert.ToBase64String(serializedData);

        obj.WriteTo(writer);
    }

    /// <summary>
    /// Reads the JSON representation of the Vector&lt;T&gt; object.
    /// </summary>
    /// <param name="reader">The JsonReader to read from.</param>
    /// <param name="objectType">The type of the object to convert.</param>
    /// <param name="existingValue">The existing value of the object being read.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    /// <returns>The Vector&lt;T&gt; object deserialized from JSON.</returns>
    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null)
            return null;

        JObject obj = JObject.Load(reader);

        int length = obj["Length"]?.Value<int>() ?? 0;

        if (length == 0)
            return Vector<T>.Empty();

        string? base64Data = obj["Data"]?.Value<string>();
        if (string.IsNullOrEmpty(base64Data))
            return new Vector<T>(length);

        // Decode and deserialize using our existing helper
        byte[] binaryData = Convert.FromBase64String(base64Data);

        // Use the existing deserializer
        return SerializationHelper<T>.DeserializeVector(binaryData);
    }
}