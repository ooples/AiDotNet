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
    /// <param name="component">The component performing the validation (optional).</param>
    /// <param name="operation">The operation being performed (optional).</param>
    public static void ValidateWriter(BinaryWriter? writer, string component = "", string operation = "")
    {
        if (writer == null)
        {
            var (resolvedComponent, resolvedOperation) = ValidationHelper<object>.ResolveCallerInfo(component, operation);
            throw new SerializationException("Binary writer cannot be null", resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a binary reader is not null.
    /// </summary>
    /// <param name="reader">The binary reader to validate.</param>
    /// <param name="component">The component performing the validation (optional).</param>
    /// <param name="operation">The operation being performed (optional).</param>
    public static void ValidateReader(BinaryReader? reader, string component = "", string operation = "")
    {
        if (reader == null)
        {
            var (resolvedComponent, resolvedOperation) = ValidationHelper<object>.ResolveCallerInfo(component, operation);
            throw new SerializationException("Binary reader cannot be null", resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a stream is not null and has the expected capabilities.
    /// </summary>
    /// <param name="stream">The stream to validate.</param>
    /// <param name="requireRead">Whether the stream should be readable.</param>
    /// <param name="requireWrite">Whether the stream should be writable.</param>
    /// <param name="component">The component performing the validation (optional).</param>
    /// <param name="operation">The operation being performed (optional).</param>
    public static void ValidateStream(Stream? stream, bool requireRead = false, bool requireWrite = false, string component = "", string operation = "")
    {
        var (resolvedComponent, resolvedOperation) = ValidationHelper<object>.ResolveCallerInfo(component, operation);

        if (stream == null)
        {
            throw new SerializationException("Stream cannot be null", resolvedComponent, resolvedOperation);
        }

        if (requireRead && !stream.CanRead)
        {
            throw new SerializationException("Stream must be readable", resolvedComponent, resolvedOperation);
        }

        if (requireWrite && !stream.CanWrite)
        {
            throw new SerializationException("Stream must be writable", resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a file path is not null or empty.
    /// </summary>
    /// <param name="filePath">The file path to validate.</param>
    /// <param name="component">The component performing the validation (optional).</param>
    /// <param name="operation">The operation being performed (optional).</param>
    public static void ValidateFilePath(string? filePath, string component = "", string operation = "")
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            var (resolvedComponent, resolvedOperation) = ValidationHelper<object>.ResolveCallerInfo(component, operation);
            throw new SerializationException("File path cannot be null or empty", resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a serialized version matches the expected version.
    /// </summary>
    /// <param name="actualVersion">The actual version from the serialized data.</param>
    /// <param name="expectedVersion">The expected version for the current implementation.</param>
    /// <param name="component">The component performing the validation (optional).</param>
    /// <param name="operation">The operation being performed (optional).</param>
    public static void ValidateVersion(int actualVersion, int expectedVersion, string component = "", string operation = "")
    {
        if (actualVersion != expectedVersion)
        {
            var (resolvedComponent, resolvedOperation) = ValidationHelper<object>.ResolveCallerInfo(component, operation);
            throw new SerializationException(
                $"Version mismatch. Expected version {expectedVersion}, but found version {actualVersion}",
                resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a layer type name is not null or empty.
    /// </summary>
    /// <param name="layerTypeName">The layer type name to validate.</param>
    /// <param name="component">The component performing the validation (optional).</param>
    /// <param name="operation">The operation being performed (optional).</param>
    /// <exception cref="SerializationException">Thrown when the layer type name is null or empty.</exception>
    public static void ValidateLayerTypeName(string? layerTypeName, string component = "", string operation = "")
    {
        if (string.IsNullOrEmpty(layerTypeName))
        {
            var (resolvedComponent, resolvedOperation) = ValidationHelper<object>.ResolveCallerInfo(component, operation);
            throw new SerializationException(
                "Layer type name cannot be null or empty during serialization/deserialization.",
                resolvedComponent,
                resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a layer type exists and can be instantiated.
    /// </summary>
    /// <param name="layerTypeName">The name of the layer type.</param>
    /// <param name="layerType">The resolved layer type.</param>
    /// <param name="component">The component performing the validation (optional).</param>
    /// <param name="operation">The operation being performed (optional).</param>
    /// <exception cref="SerializationException">Thrown when the layer type cannot be found.</exception>
    public static void ValidateLayerTypeExists(string layerTypeName, Type? layerType, string component = "", string operation = "")
    {
        if (layerType == null)
        {
            var (resolvedComponent, resolvedOperation) = ValidationHelper<object>.ResolveCallerInfo(component, operation);
            throw new SerializationException(
                $"Could not find layer type '{layerTypeName}'. Make sure the assembly containing this type is loaded.",
                resolvedComponent,
                resolvedOperation);
        }
    }
}
