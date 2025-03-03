namespace AiDotNet.Validation;

/// <summary>
/// Provides validation methods for serialization and deserialization operations.
/// </summary>
public static class SerializationValidator
{
    /// <summary>
    /// Validates that a binary writer is not null.
    /// </summary>
    /// <param name="writer">The binary writer to validate.</param>
    /// <param name="component">The component performing the validation.</param>
    public static void ValidateWriter(BinaryWriter? writer, string component)
    {
        if (writer == null)
        {
            throw new SerializationException("BinaryWriter cannot be null", component, "Serialize");
        }
    }

    /// <summary>
    /// Validates that a binary reader is not null.
    /// </summary>
    /// <param name="reader">The binary reader to validate.</param>
    /// <param name="component">The component performing the validation.</param>
    public static void ValidateReader(BinaryReader? reader, string component)
    {
        if (reader == null)
        {
            throw new SerializationException("BinaryReader cannot be null", component, "Deserialize");
        }
    }

    /// <summary>
    /// Validates that a layer type name is valid.
    /// </summary>
    /// <param name="layerTypeName">The layer type name to validate.</param>
    /// <param name="component">The component performing the validation.</param>
    public static void ValidateLayerTypeName(string? layerTypeName, string component)
    {
        if (string.IsNullOrEmpty(layerTypeName))
        {
            throw new SerializationException("Encountered an empty layer type name during deserialization", component, "Deserialize");
        }
    }

    /// <summary>
    /// Validates that a layer type can be found.
    /// </summary>
    /// <param name="layerTypeName">The layer type name to validate.</param>
    /// <param name="layerType">The resolved layer type.</param>
    /// <param name="component">The component performing the validation.</param>
    public static void ValidateLayerTypeExists(string layerTypeName, Type? layerType, string component)
    {
        if (layerType == null)
        {
            throw new SerializationException($"Cannot find type {layerTypeName}", component, "Deserialize");
        }
    }
}