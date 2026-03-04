using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;

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
    /// Gets the dynamic shape information describing which dimensions are variable.
    /// </summary>
    public DynamicShapeInfo DynamicShapeInfo { get; }

    /// <summary>
    /// Gets the encryption scheme applied to the payload.
    /// </summary>
    public PayloadEncryptionScheme EncryptionScheme { get; }

    /// <summary>
    /// Gets the PBKDF2 salt used for key derivation when the payload is encrypted, or null if unencrypted.
    /// </summary>
    public byte[]? Salt { get; }

    /// <summary>
    /// Gets the AES-GCM nonce (IV) when the payload is encrypted, or null if unencrypted.
    /// </summary>
    public byte[]? Nonce { get; }

    /// <summary>
    /// Gets the AES-GCM authentication tag when the payload is encrypted, or null if unencrypted.
    /// </summary>
    public byte[]? Tag { get; }

    /// <summary>
    /// Gets whether the payload is encrypted and requires a license key to load.
    /// </summary>
    public bool IsEncrypted => EncryptionScheme != PayloadEncryptionScheme.None;

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
        int headerLength,
        DynamicShapeInfo? dynamicShapeInfo = null,
        PayloadEncryptionScheme encryptionScheme = PayloadEncryptionScheme.None,
        byte[]? salt = null,
        byte[]? nonce = null,
        byte[]? tag = null)
    {
        EnvelopeVersion = envelopeVersion;
        Format = format;
        TypeName = typeName ?? string.Empty;
        AssemblyQualifiedName = assemblyQualifiedName ?? string.Empty;
        InputShape = inputShape ?? Array.Empty<int>();
        OutputShape = outputShape ?? Array.Empty<int>();
        PayloadLength = payloadLength;
        HeaderLength = headerLength;
        DynamicShapeInfo = dynamicShapeInfo ?? DynamicShapeInfo.None;
        EncryptionScheme = encryptionScheme;
        Salt = salt;
        Nonce = nonce;
        Tag = tag;
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
/// The envelope format (v1) is:
/// <code>
/// [Magic: 4 bytes "AIMF"]
/// [Envelope version: int32]
/// [Serialization format: int32]
/// [Type name: length-prefixed string]
/// [Assembly-qualified name: length-prefixed string]
/// [Input shape rank: int32] [Input shape dims: int32[]]
/// [Output shape rank: int32] [Output shape dims: int32[]]
/// [Dynamic input dim count: int32] [Dynamic input dim indices: int32[]]
/// [Dynamic output dim count: int32] [Dynamic output dim indices: int32[]]
/// [Encryption scheme: int32]
/// [If encrypted: Salt 16 bytes, Nonce 12 bytes, Tag 16 bytes]
/// [Payload length: int64]
/// [Payload: byte[] - plaintext or AES-256-GCM ciphertext]
/// </code>
///
/// The existing Serialize() output is stored unchanged as the payload (or encrypted).
/// The header is always plaintext, allowing Inspect() to read metadata without a key.
/// </remarks>
public static class ModelFileHeader
{
    /// <summary>
    /// Magic bytes identifying an AIMF envelope: 0x41 0x49 0x4D 0x46 = "AIMF".
    /// </summary>
    public const int AimfMagic = 0x41494D46;

    /// <summary>
    /// Current envelope version. Version 1 includes dynamic shapes and encryption support.
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
    /// <param name="dynamicShapeInfo">Optional dynamic shape information. If null, no dynamic dimensions are written.</param>
    /// <returns>A byte array containing the AIMF header followed by the payload.</returns>
    public static byte[] WrapWithHeader(
        byte[] payload,
        IModelSerializer model,
        int[] inputShape,
        int[] outputShape,
        SerializationFormat format,
        DynamicShapeInfo? dynamicShapeInfo = null)
    {
        return WrapWithHeaderInternal(
            payload, model, inputShape, outputShape, format, dynamicShapeInfo,
            PayloadEncryptionScheme.None, null, null, null);
    }

    /// <summary>
    /// Wraps serialized model data with an AIMF envelope header that includes encryption metadata.
    /// The payload should already be encrypted before calling this method.
    /// </summary>
    /// <param name="encryptedPayload">The AES-256-GCM encrypted model data.</param>
    /// <param name="model">The model instance, used to extract type information.</param>
    /// <param name="inputShape">The input shape of the model.</param>
    /// <param name="outputShape">The output shape of the model.</param>
    /// <param name="format">The serialization format of the original (pre-encryption) payload.</param>
    /// <param name="salt">The 16-byte PBKDF2 salt used for key derivation.</param>
    /// <param name="nonce">The 12-byte AES-GCM nonce.</param>
    /// <param name="tag">The 16-byte AES-GCM authentication tag.</param>
    /// <param name="dynamicShapeInfo">Optional dynamic shape information.</param>
    /// <returns>A byte array containing the AIMF header followed by the encrypted payload.</returns>
    public static byte[] WrapWithHeaderEncrypted(
        byte[] encryptedPayload,
        IModelSerializer model,
        int[] inputShape,
        int[] outputShape,
        SerializationFormat format,
        byte[] salt,
        byte[] nonce,
        byte[] tag,
        DynamicShapeInfo? dynamicShapeInfo = null)
    {
        if (salt is null)
        {
            throw new ArgumentNullException(nameof(salt));
        }

        if (salt.Length != 16)
        {
            throw new ArgumentException($"Salt must be exactly 16 bytes, got {salt.Length}.", nameof(salt));
        }

        if (nonce is null)
        {
            throw new ArgumentNullException(nameof(nonce));
        }

        if (nonce.Length != 12)
        {
            throw new ArgumentException($"Nonce must be exactly 12 bytes, got {nonce.Length}.", nameof(nonce));
        }

        if (tag is null)
        {
            throw new ArgumentNullException(nameof(tag));
        }

        if (tag.Length != 16)
        {
            throw new ArgumentException($"Tag must be exactly 16 bytes, got {tag.Length}.", nameof(tag));
        }

        return WrapWithHeaderInternal(
            encryptedPayload, model, inputShape, outputShape, format, dynamicShapeInfo,
            PayloadEncryptionScheme.AesGcm256, salt, nonce, tag);
    }

    /// <summary>
    /// Wraps an encrypted payload with an AIMF header using a specified encryption scheme.
    /// </summary>
    public static byte[] WrapWithHeaderEncrypted(
        byte[] encryptedPayload,
        IModelSerializer model,
        int[] inputShape,
        int[] outputShape,
        SerializationFormat format,
        byte[] salt,
        byte[] nonce,
        byte[] tag,
        PayloadEncryptionScheme scheme,
        DynamicShapeInfo? dynamicShapeInfo = null)
    {
        if (salt is null)
        {
            throw new ArgumentNullException(nameof(salt));
        }

        if (salt.Length != 16)
        {
            throw new ArgumentException($"Salt must be exactly 16 bytes, got {salt.Length}.", nameof(salt));
        }

        if (nonce is null)
        {
            throw new ArgumentNullException(nameof(nonce));
        }

        if (nonce.Length != 12)
        {
            throw new ArgumentException($"Nonce must be exactly 12 bytes, got {nonce.Length}.", nameof(nonce));
        }

        if (tag is null)
        {
            throw new ArgumentNullException(nameof(tag));
        }

        if (tag.Length != 16)
        {
            throw new ArgumentException($"Tag must be exactly 16 bytes, got {tag.Length}.", nameof(tag));
        }

        return WrapWithHeaderInternal(
            encryptedPayload, model, inputShape, outputShape, format, dynamicShapeInfo,
            scheme, salt, nonce, tag);
    }

    private static byte[] WrapWithHeaderInternal(
        byte[] payload,
        IModelSerializer model,
        int[] inputShape,
        int[] outputShape,
        SerializationFormat format,
        DynamicShapeInfo? dynamicShapeInfo,
        PayloadEncryptionScheme encryptionScheme,
        byte[]? salt,
        byte[]? nonce,
        byte[]? tag)
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

        // Dynamic shape dimensions
        var dynInfo = dynamicShapeInfo ?? DynamicShapeInfo.None;
        writer.Write(dynInfo.DynamicInputDimensions.Length);
        for (int i = 0; i < dynInfo.DynamicInputDimensions.Length; i++)
        {
            writer.Write(dynInfo.DynamicInputDimensions[i]);
        }

        writer.Write(dynInfo.DynamicOutputDimensions.Length);
        for (int i = 0; i < dynInfo.DynamicOutputDimensions.Length; i++)
        {
            writer.Write(dynInfo.DynamicOutputDimensions[i]);
        }

        // Encryption scheme
        writer.Write((int)encryptionScheme);

        if (encryptionScheme != PayloadEncryptionScheme.None)
        {
            // Salt (16 bytes), Nonce (12 bytes), Tag (16 bytes)
            writer.Write(salt ?? throw new InvalidOperationException("Salt is required for encrypted payloads."));
            writer.Write(nonce ?? throw new InvalidOperationException("Nonce is required for encrypted payloads."));
            writer.Write(tag ?? throw new InvalidOperationException("Tag is required for encrypted payloads."));
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
                "This is not a valid AIMF model file. " +
                "Models must be saved with the AIMF envelope format.");
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

        // Serialization format (validate against known values)
        int formatRaw = reader.ReadInt32();
        if (!Enum.IsDefined(typeof(SerializationFormat), formatRaw))
        {
            throw new InvalidOperationException(
                $"AIMF header contains unknown serialization format value {formatRaw}.");
        }
        var format = (SerializationFormat)formatRaw;

        // Type name
        string typeName = reader.ReadString();

        // Assembly-qualified name
        string assemblyQualifiedName = reader.ReadString();

        // Input shape (bounded to prevent DoS from malicious files)
        const int MaxShapeRank = 32;
        int inputRank = reader.ReadInt32();
        if (inputRank < 0 || inputRank > MaxShapeRank)
        {
            throw new InvalidOperationException(
                $"AIMF header has invalid input shape rank {inputRank}. Maximum supported rank is {MaxShapeRank}.");
        }

        var inputShape = new int[inputRank];
        for (int i = 0; i < inputRank; i++)
        {
            inputShape[i] = reader.ReadInt32();
        }

        // Output shape
        int outputRank = reader.ReadInt32();
        if (outputRank < 0 || outputRank > MaxShapeRank)
        {
            throw new InvalidOperationException(
                $"AIMF header has invalid output shape rank {outputRank}. Maximum supported rank is {MaxShapeRank}.");
        }

        var outputShape = new int[outputRank];
        for (int i = 0; i < outputRank; i++)
        {
            outputShape[i] = reader.ReadInt32();
        }

        // Dynamic shape dimensions
        int dynInputCount = reader.ReadInt32();
        if (dynInputCount < 0 || dynInputCount > MaxShapeRank)
        {
            throw new InvalidOperationException(
                $"AIMF header has invalid dynamic input dimension count {dynInputCount}.");
        }

        var dynInputDims = new int[dynInputCount];
        for (int i = 0; i < dynInputCount; i++)
        {
            dynInputDims[i] = reader.ReadInt32();
        }

        int dynOutputCount = reader.ReadInt32();
        if (dynOutputCount < 0 || dynOutputCount > MaxShapeRank)
        {
            throw new InvalidOperationException(
                $"AIMF header has invalid dynamic output dimension count {dynOutputCount}.");
        }

        var dynOutputDims = new int[dynOutputCount];
        for (int i = 0; i < dynOutputCount; i++)
        {
            dynOutputDims[i] = reader.ReadInt32();
        }

        var dynamicShapeInfo = new DynamicShapeInfo
        {
            DynamicInputDimensions = dynInputDims,
            DynamicOutputDimensions = dynOutputDims
        };

        // Encryption scheme (validate against known values)
        int encryptionRaw = reader.ReadInt32();
        if (!Enum.IsDefined(typeof(PayloadEncryptionScheme), encryptionRaw))
        {
            throw new InvalidOperationException(
                $"AIMF header contains unknown encryption scheme value {encryptionRaw}.");
        }
        var encryptionScheme = (PayloadEncryptionScheme)encryptionRaw;
        byte[]? salt = null;
        byte[]? nonce = null;
        byte[]? tag = null;

        if (encryptionScheme != PayloadEncryptionScheme.None)
        {
            salt = reader.ReadBytes(16);
            nonce = reader.ReadBytes(12);
            tag = reader.ReadBytes(16);
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
            headerLength,
            dynamicShapeInfo,
            encryptionScheme,
            salt,
            nonce,
            tag);
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
