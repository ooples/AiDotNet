using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Helpers;

/// <summary>
/// Contains parsed metadata from an AIMF (AI Model File) envelope header.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When a model file is saved with the AIMF envelope, the header contains
/// metadata that describes the model without needing to load it fully. This record holds
/// all the information extracted from that header.
/// </remarks>
public sealed class ModelFileInfo
{
    /// <summary>
    /// Gets the envelope format version.
    /// </summary>
    public int EnvelopeVersion { get; }

    /// <summary>
    /// Gets the serialization format of the model payload.
    /// </summary>
    public SerializationFormat Format { get; }

    /// <summary>
    /// Gets the short type name of the model (e.g., "ConvolutionalNeuralNetwork`1").
    /// </summary>
    public string TypeName { get; }

    /// <summary>
    /// Gets the assembly-qualified type name for fallback resolution.
    /// </summary>
    public string AssemblyQualifiedName { get; }

    /// <summary>
    /// Gets the input shape of the model, or empty array if not available.
    /// </summary>
    public int[] InputShape { get; }

    /// <summary>
    /// Gets the output shape of the model, or empty array if not available.
    /// </summary>
    public int[] OutputShape { get; }

    /// <summary>
    /// Gets the length of the model payload in bytes.
    /// </summary>
    public long PayloadLength { get; }

    /// <summary>
    /// Gets the byte offset where the payload starts in the original data.
    /// </summary>
    public int HeaderLength { get; }

    /// <summary>
    /// Creates a new ModelFileInfo with the specified header values.
    /// </summary>
    public ModelFileInfo(
        int envelopeVersion,
        SerializationFormat format,
        string typeName,
        string assemblyQualifiedName,
        int[] inputShape,
        int[] outputShape,
        long payloadLength,
        int headerLength)
    {
        EnvelopeVersion = envelopeVersion;
        Format = format;
        TypeName = typeName ?? string.Empty;
        AssemblyQualifiedName = assemblyQualifiedName ?? string.Empty;
        InputShape = inputShape ?? Array.Empty<int>();
        OutputShape = outputShape ?? Array.Empty<int>();
        PayloadLength = payloadLength;
        HeaderLength = headerLength;
    }
}

/// <summary>
/// Provides methods for reading and writing the AIMF (AI Model File) binary envelope header.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When you save a machine learning model, this helper wraps the model data
/// with a small header that describes what type of model it is, its input/output shapes, and
/// what format the data is in. This allows tools to identify and load models automatically
/// without needing to know the model type in advance.
///
/// The envelope format is:
/// <code>
/// [Magic: 4 bytes "AIMF"]
/// [Envelope version: int32]
/// [Serialization format: int32]
/// [Type name: length-prefixed string]
/// [Assembly-qualified name: length-prefixed string]
/// [Input shape rank: int32] [Input shape dims: int32[]]
/// [Output shape rank: int32] [Output shape dims: int32[]]
/// [Payload length: int64]
/// [Payload: byte[]]
/// </code>
///
/// The existing Serialize() output is stored unchanged as the payload.
/// This preserves full backward compatibility with all existing model implementations.
/// </remarks>
public static class ModelFileHeader
{
    /// <summary>
    /// Magic bytes identifying an AIMF envelope: 0x41 0x49 0x4D 0x46 = "AIMF".
    /// </summary>
    public const int AimfMagic = 0x41494D46;

    /// <summary>
    /// Current envelope version.
    /// </summary>
    public const int CurrentEnvelopeVersion = 1;

    /// <summary>
    /// Wraps serialized model data with an AIMF envelope header.
    /// </summary>
    /// <param name="payload">The serialized model data from Serialize().</param>
    /// <param name="model">The model instance, used to extract type information.</param>
    /// <param name="inputShape">The input shape of the model. Pass empty array if unknown.</param>
    /// <param name="outputShape">The output shape of the model. Pass empty array if unknown.</param>
    /// <param name="format">The serialization format of the payload.</param>
    /// <returns>A byte array containing the AIMF header followed by the payload.</returns>
    public static byte[] WrapWithHeader(
        byte[] payload,
        IModelSerializer model,
        int[] inputShape,
        int[] outputShape,
        SerializationFormat format)
    {
        if (payload is null)
        {
            throw new ArgumentNullException(nameof(payload));
        }

        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        inputShape ??= Array.Empty<int>();
        outputShape ??= Array.Empty<int>();

        var modelType = model.GetType();
        string typeName = modelType.Name;
        string assemblyQualifiedName = modelType.AssemblyQualifiedName ?? modelType.FullName ?? typeName;

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Magic
        writer.Write(AimfMagic);

        // Envelope version
        writer.Write(CurrentEnvelopeVersion);

        // Serialization format
        writer.Write((int)format);

        // Type name (length-prefixed via BinaryWriter.Write(string))
        writer.Write(typeName);

        // Assembly-qualified name
        writer.Write(assemblyQualifiedName);

        // Input shape: rank + dims
        writer.Write(inputShape.Length);
        for (int i = 0; i < inputShape.Length; i++)
        {
            writer.Write(inputShape[i]);
        }

        // Output shape: rank + dims
        writer.Write(outputShape.Length);
        for (int i = 0; i < outputShape.Length; i++)
        {
            writer.Write(outputShape[i]);
        }

        // Payload length
        writer.Write((long)payload.Length);

        // Payload
        writer.Write(payload);

        writer.Flush();
        return ms.ToArray();
    }

    /// <summary>
    /// Checks whether the given data starts with the AIMF magic bytes.
    /// </summary>
    /// <param name="data">The data to check.</param>
    /// <returns>True if the data has an AIMF envelope header.</returns>
    public static bool HasHeader(byte[] data)
    {
        if (data is null || data.Length < 4)
        {
            return false;
        }

        int magic = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
        return magic == AimfMagic;
    }

    /// <summary>
    /// Checks whether a file starts with the AIMF magic bytes by reading only the first 4 bytes.
    /// </summary>
    /// <param name="filePath">The path to the file to check.</param>
    /// <returns>True if the file has an AIMF envelope header.</returns>
    public static bool HasHeader(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
        {
            return false;
        }

        var buffer = new byte[4];
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (fs.Read(buffer, 0, 4) < 4)
        {
            return false;
        }

        return HasHeader(buffer);
    }

    /// <summary>
    /// Reads the AIMF envelope header from a byte array.
    /// </summary>
    /// <param name="data">The data containing the AIMF envelope.</param>
    /// <returns>A ModelFileInfo containing the parsed header information.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the data does not contain a valid AIMF header.</exception>
    public static ModelFileInfo ReadHeader(byte[] data)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (!HasHeader(data))
        {
            throw new InvalidOperationException(
                "Data does not contain a valid AIMF envelope header. " +
                "This file may have been saved without the AIMF envelope (legacy format). " +
                "Use the model's LoadModel() method directly for legacy files.");
        }

        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Magic (already validated)
        reader.ReadInt32();

        // Envelope version
        int envelopeVersion = reader.ReadInt32();
        if (envelopeVersion > CurrentEnvelopeVersion)
        {
            throw new InvalidOperationException(
                $"AIMF envelope version {envelopeVersion} is not supported. " +
                $"Maximum supported version is {CurrentEnvelopeVersion}. " +
                "Please update AiDotNet to the latest version.");
        }

        // Serialization format
        var format = (SerializationFormat)reader.ReadInt32();

        // Type name
        string typeName = reader.ReadString();

        // Assembly-qualified name
        string assemblyQualifiedName = reader.ReadString();

        // Input shape
        int inputRank = reader.ReadInt32();
        var inputShape = new int[inputRank];
        for (int i = 0; i < inputRank; i++)
        {
            inputShape[i] = reader.ReadInt32();
        }

        // Output shape
        int outputRank = reader.ReadInt32();
        var outputShape = new int[outputRank];
        for (int i = 0; i < outputRank; i++)
        {
            outputShape[i] = reader.ReadInt32();
        }

        // Payload length
        long payloadLength = reader.ReadInt64();

        int headerLength = (int)ms.Position;

        return new ModelFileInfo(
            envelopeVersion,
            format,
            typeName,
            assemblyQualifiedName,
            inputShape,
            outputShape,
            payloadLength,
            headerLength);
    }

    /// <summary>
    /// Extracts the model payload from data that has an AIMF envelope header.
    /// </summary>
    /// <param name="data">The full data including the AIMF header.</param>
    /// <param name="info">The parsed header info from ReadHeader. If null, the header will be read first.</param>
    /// <returns>The payload bytes (the original Serialize() output).</returns>
    public static byte[] ExtractPayload(byte[] data, ModelFileInfo? info = null)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        info ??= ReadHeader(data);

        var payload = new byte[info.PayloadLength];
        Array.Copy(data, info.HeaderLength, payload, 0, (int)info.PayloadLength);
        return payload;
    }
}
