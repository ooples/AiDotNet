namespace AiDotNet.Serialization;

/// <summary>
/// Custom JSON converter for interface types to enable proper serialization and deserialization.
/// </summary>
/// <remarks>
/// <para>
/// This converter handles the serialization and deserialization of interfaces by storing the concrete
/// type information along with the object data. It follows the same pattern as the binary serialization
/// methods SerializeInterface and DeserializeInterface in the SerializationHelper class.
/// </para>
/// <para><b>For Beginners:</b> This converter ensures that interfaces (like IFullModel) can be correctly saved and loaded.
/// It stores the actual type of the object along with its data, so when loading it later, we know exactly what
/// type of object to create.
/// </para>
/// </remarks>
public class InterfaceJsonConverter : JsonConverter
{
    /// <summary>
    /// Determines whether this converter can convert the specified object type.
    /// </summary>
    /// <param name="objectType">The type of the object to check.</param>
    /// <returns>True if this converter can convert the specified type; otherwise, false.</returns>
    public override bool CanConvert(Type objectType)
    {
        // Only handle interfaces or abstract classes
        return objectType.IsInterface || (objectType.IsAbstract && !objectType.IsSealed);
    }

    /// <summary>
    /// Writes the JSON representation of the interface object.
    /// </summary>
    /// <param name="writer">The JsonWriter to write to.</param>
    /// <param name="value">The interface object to convert to JSON.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        // Following the pattern in SerializeInterface method
        if (value == null)
        {
            writer.WriteNull();
            return;
        }

        // Get the concrete type information
        Type concreteType = value.GetType();

        // Create a container with both type and object data
        JObject container = new JObject();
        container["TypeName"] = concreteType.AssemblyQualifiedName;

        // Serialize the actual object data
        JToken objectData = JToken.FromObject(value, serializer);
        container["ObjectData"] = objectData;

        container.WriteTo(writer);
    }

    /// <summary>
    /// Reads the JSON representation of the interface object.
    /// </summary>
    /// <param name="reader">The JsonReader to read from.</param>
    /// <param name="objectType">The type of the object to convert.</param>
    /// <param name="existingValue">The existing value of the object being read.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    /// <returns>The interface object deserialized from JSON.</returns>
    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        // Following the pattern in DeserializeInterface method
        if (reader.TokenType == JsonToken.Null)
            return null;

        // Read the JSON container object
        JObject container = JObject.Load(reader);

        // Get the type name
        string? typeName = container["TypeName"]?.Value<string>();
        if (string.IsNullOrEmpty(typeName))
        {
            return null;
        }

        // Resolve the type
        Type? concreteType = Type.GetType(typeName);
        if (concreteType == null)
        {
            throw new InvalidOperationException($"Cannot find type {typeName}");
        }

        // Validate the type implements the expected interface
        if (!objectType.IsAssignableFrom(concreteType))
        {
            throw new InvalidOperationException($"Type {typeName} does not implement interface {objectType.Name}");
        }

        // Get the object data
        JToken? objectData = container["ObjectData"];
        if (objectData == null)
        {
            // If no object data, create a default instance
            return Activator.CreateInstance(concreteType)
                ?? throw new InvalidOperationException($"Failed to create instance of type {typeName}");
        }

        // Deserialize the object data to the concrete type
        return objectData.ToObject(concreteType, serializer)
            ?? throw new InvalidOperationException($"Failed to deserialize object of type {typeName}");
    }
}