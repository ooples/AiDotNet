namespace AiDotNet.Enums;

/// <summary>
/// Specifies the serialization format used for the model payload within an AIMF envelope.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When a model is saved to disk, its internal data can be stored in different formats.
/// This enum tells the loader what format to expect inside the file so it can correctly read the data.
///
/// The AIMF envelope header always uses a binary format, but the actual model data inside can vary:
/// - Binary: Raw bytes written with BinaryWriter (most neural networks)
/// - Json: JSON-serialized text (clustering models, some statistical models)
/// - HybridBinary: A mix of binary and structured data (some complex models)
/// </remarks>
public enum SerializationFormat
{
    /// <summary>
    /// Model data is stored as raw binary bytes using BinaryWriter.
    /// This is the most common format for neural networks.
    /// </summary>
    Binary = 0,

    /// <summary>
    /// Model data is stored as JSON text.
    /// Used by clustering models and some statistical models.
    /// </summary>
    Json = 1,

    /// <summary>
    /// Model data uses a hybrid format combining binary and structured data.
    /// Used by some complex models that mix binary weights with structured metadata.
    /// </summary>
    HybridBinary = 2
}
