using System.Security.Cryptography;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides static methods for loading self-describing AIMF model files with automatic type detection.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When you save a model using the new AIMF envelope format, the file contains
/// metadata about what type of model it is. ModelLoader reads this metadata and automatically creates
/// the correct model instance, so you don't need to know the exact model type in advance.
///
/// Example usage:
/// <code>
/// // Save a model (automatically adds AIMF header)
/// myNetwork.SaveModel("model.aimf");
///
/// // Load it back - ModelLoader figures out the type automatically
/// var model = ModelLoader.Load&lt;double&gt;("model.aimf");
///
/// // Inspect a model file without loading it
/// var info = ModelLoader.Inspect("model.aimf");
/// // Result is available in the returned value
/// // Result is available in the returned value
/// </code>
///
/// All model files must use the AIMF envelope format. Files saved with SaveModel()
/// automatically include the AIMF header.
/// </remarks>
public static class ModelLoader
{
    /// <summary>
    /// Loads a self-describing AIMF model file, automatically detecting and instantiating the correct model type.
    /// </summary>
    /// <typeparam name="T">
    /// The numeric type used by the model (e.g., double, float).
    /// This must match the type the model was originally trained with.
    /// </typeparam>
    /// <param name="filePath">The path to the AIMF model file.</param>
    /// <param name="licenseKey">
    /// Optional license key for encrypted models. If the model is encrypted and no key is provided,
    /// an <see cref="InvalidOperationException"/> is thrown.
    /// </param>
    /// <returns>The deserialized model instance.</returns>
    /// <exception cref="ArgumentException">Thrown when the file path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the file does not have an AIMF header, the model type cannot be resolved,
    /// or the model is encrypted and no license key is provided.
    /// </exception>
    /// <exception cref="CryptographicException">
    /// Thrown when the license key is incorrect or the encrypted data has been tampered with.
    /// </exception>
    public static IModelSerializer Load<T>(string filePath, string? licenseKey = null, byte[]? decryptionToken = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);
        }

        byte[] data = File.ReadAllBytes(filePath);
        return LoadFromBytes<T>(data, licenseKey, decryptionToken);
    }

    /// <summary>
    /// Loads a self-describing AIMF model from a byte array.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="data">The byte array containing the AIMF-enveloped model.</param>
    /// <param name="licenseKey">
    /// Optional license key for encrypted models. If the model is encrypted and no key is provided,
    /// an <see cref="InvalidOperationException"/> is thrown.
    /// </param>
    /// <returns>The deserialized model instance.</returns>
    /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the data does not have an AIMF header, the model type cannot be resolved,
    /// or the model is encrypted and no license key is provided.
    /// </exception>
    /// <exception cref="CryptographicException">
    /// Thrown when the license key is incorrect or the encrypted data has been tampered with.
    /// </exception>
    public static IModelSerializer LoadFromBytes<T>(byte[] data, string? licenseKey = null, byte[]? decryptionToken = null)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (!ModelFileHeader.HasHeader(data))
        {
            throw new InvalidOperationException(
                "The data does not contain an AIMF envelope header. " +
                "This is not a valid AIMF model file. " +
                "Models must be saved with the AIMF envelope format.");
        }

        // Read header
        var info = ModelFileHeader.ReadHeader(data);

        // Resolve the model type
        var openGenericType = ModelTypeRegistry.Resolve(info.TypeName, info.AssemblyQualifiedName);
        if (openGenericType is null)
        {
            throw new InvalidOperationException(
                $"Cannot resolve model type '{info.TypeName}'. " +
                $"Assembly-qualified name: '{info.AssemblyQualifiedName}'. " +
                "The model type may not be available in the current assembly. " +
                "Register external model types using ModelTypeRegistry.Register() or ModelTypeRegistry.RegisterAssembly().");
        }

        // Create an instance
        var model = ModelTypeRegistry.CreateInstance<T>(openGenericType);

        // Extract payload (raw bytes, possibly encrypted)
        byte[] payload = ModelFileHeader.ExtractPayload(data, info);

        // Handle encryption
        if (info.IsEncrypted)
        {
            if (licenseKey is null || string.IsNullOrWhiteSpace(licenseKey))
            {
                throw new InvalidOperationException(
                    "This model is encrypted. Provide a license key to load it.");
            }

            byte[] salt = info.Salt ?? throw new InvalidOperationException(
                "Encrypted AIMF file is missing required salt parameter.");
            byte[] nonce = info.Nonce ?? throw new InvalidOperationException(
                "Encrypted AIMF file is missing required nonce parameter.");
            byte[] tag = info.Tag ?? throw new InvalidOperationException(
                "Encrypted AIMF file is missing required tag parameter.");

            var aad = ModelPayloadEncryption.BuildAad(info.TypeName, info.InputShape, info.OutputShape);

            if (info.EncryptionScheme == PayloadEncryptionScheme.AesGcm256Signed)
            {
                payload = ModelPayloadEncryption.DecryptSigned(payload, licenseKey, salt, nonce, tag, aad, decryptionToken);
            }
            else if (info.EncryptionScheme == PayloadEncryptionScheme.AesGcm256)
            {
                payload = ModelPayloadEncryption.Decrypt(payload, licenseKey, salt, nonce, tag, aad);
            }
            else
            {
                throw new InvalidOperationException(
                    $"Unknown encryption scheme '{info.EncryptionScheme}'. " +
                    "Cannot decrypt this model. Please update AiDotNet to the latest version.");
            }
        }

        model.Deserialize(payload);

        // Optional shape validation if the model implements IModelShape
        if (model is IModelShape shapeModel)
        {
            ValidateShapes(shapeModel, info);
        }

        return model;
    }

    /// <summary>
    /// Saves a model to an encrypted AIMF file that requires a license key to load.
    /// </summary>
    /// <param name="model">The model to save. Must implement <see cref="IModelSerializer"/>.</param>
    /// <param name="filePath">The output file path.</param>
    /// <param name="licenseKey">The license key used to encrypt the model weights.</param>
    /// <param name="inputShape">The input shape of the model. Pass empty array if unknown.</param>
    /// <param name="outputShape">The output shape of the model. Pass empty array if unknown.</param>
    /// <param name="format">The serialization format of the payload.</param>
    /// <param name="dynamicShapeInfo">Optional dynamic shape information.</param>
    /// <exception cref="ArgumentNullException">Thrown when model is null.</exception>
    /// <exception cref="ArgumentException">Thrown when filePath or licenseKey is null/empty.</exception>
    /// <exception cref="PlatformNotSupportedException">Thrown on .NET Framework 4.7.1.</exception>
    public static void SaveEncrypted(
        IModelSerializer model,
        string filePath,
        string licenseKey,
        int[] inputShape,
        int[] outputShape,
        SerializationFormat format = SerializationFormat.Binary,
        DynamicShapeInfo? dynamicShapeInfo = null,
        byte[]? decryptionToken = null)
    {
        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (string.IsNullOrWhiteSpace(licenseKey))
        {
            throw new ArgumentException("License key cannot be null or empty.", nameof(licenseKey));
        }

        inputShape ??= Array.Empty<int>();
        outputShape ??= Array.Empty<int>();

        // Serialize the model
        byte[] plaintext = model.Serialize();

        // Build AAD from model metadata
        var modelType = model.GetType();
        string typeName = modelType.Name;
        var aad = ModelPayloadEncryption.BuildAad(typeName, inputShape, outputShape);

        // Choose encryption scheme based on whether this is an official build
        EncryptedPayload encrypted;
        PayloadEncryptionScheme scheme;

        if (BuildKeyProvider.IsOfficialBuild)
        {
            encrypted = ModelPayloadEncryption.EncryptSigned(plaintext, licenseKey, aad, decryptionToken);
            scheme = PayloadEncryptionScheme.AesGcm256Signed;
        }
        else
        {
            encrypted = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, aad);
            scheme = PayloadEncryptionScheme.AesGcm256;
        }

        // Wrap with header
        byte[] enveloped = ModelFileHeader.WrapWithHeaderEncrypted(
            encrypted.Ciphertext,
            model,
            inputShape,
            outputShape,
            format,
            encrypted.Salt,
            encrypted.Nonce,
            encrypted.Tag,
            scheme,
            dynamicShapeInfo);

        string? directory = Path.GetDirectoryName(filePath);
        if (directory is not null && !Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllBytes(filePath, enveloped);
    }

    /// <summary>
    /// Reads only the header of an AIMF model file without loading the full model.
    /// This is useful for inspecting model metadata (type, shapes) without the cost of full deserialization.
    /// </summary>
    /// <param name="filePath">The path to the AIMF model file.</param>
    /// <returns>A ModelFileInfo containing the header metadata.</returns>
    /// <exception cref="ArgumentException">Thrown when the file path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the file does not have an AIMF header.</exception>
    public static ModelFileInfo Inspect(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);
        }

        byte[] data = File.ReadAllBytes(filePath);
        return ModelFileHeader.ReadHeader(data);
    }

    /// <summary>
    /// Checks whether a file is a self-describing AIMF model file by reading only the first 4 bytes.
    /// </summary>
    /// <param name="filePath">The path to the file to check.</param>
    /// <returns>True if the file starts with the AIMF magic bytes.</returns>
    public static bool IsSelfDescribing(string filePath)
    {
        return ModelFileHeader.HasHeader(filePath);
    }

    /// <summary>
    /// Validates that the deserialized model's shapes match those stored in the header.
    /// Throws if shapes don't match to prevent loading corrupted or mismatched models.
    /// </summary>
    private static void ValidateShapes(IModelShape model, ModelFileInfo headerInfo)
    {
        // Only validate if header had shape info (non-empty arrays)
        if (headerInfo.InputShape.Length == 0 && headerInfo.OutputShape.Length == 0)
        {
            return;
        }

        var actualInput = model.GetInputShape();
        var actualOutput = model.GetOutputShape();

        if (headerInfo.InputShape.Length > 0 && actualInput.Length == 0)
        {
            // Header has shape info but model returned empty - shape validation cannot verify integrity.
            // This may indicate the model did not restore shape info correctly.
            System.Diagnostics.Trace.TraceWarning(
                "[AIMF] Header has input shape [{0}] but deserialized model returned empty input shape. Shape validation skipped.",
                string.Join(", ", headerInfo.InputShape));
        }
        else if (headerInfo.InputShape.Length > 0 && actualInput.Length > 0 && !ShapesMatch(actualInput, headerInfo.InputShape))
        {
            throw new InvalidOperationException(
                $"Input shape mismatch. " +
                $"Header: [{string.Join(", ", headerInfo.InputShape)}], " +
                $"Model: [{string.Join(", ", actualInput)}]. " +
                "The model file may be corrupted or was saved with a different configuration.");
        }

        if (headerInfo.OutputShape.Length > 0 && actualOutput.Length == 0)
        {
            System.Diagnostics.Trace.TraceWarning(
                "[AIMF] Header has output shape [{0}] but deserialized model returned empty output shape. Shape validation skipped.",
                string.Join(", ", headerInfo.OutputShape));
        }
        else if (headerInfo.OutputShape.Length > 0 && actualOutput.Length > 0 && !ShapesMatch(actualOutput, headerInfo.OutputShape))
        {
            throw new InvalidOperationException(
                $"Output shape mismatch. " +
                $"Header: [{string.Join(", ", headerInfo.OutputShape)}], " +
                $"Model: [{string.Join(", ", actualOutput)}]. " +
                "The model file may be corrupted or was saved with a different configuration.");
        }
    }

    private static bool ShapesMatch(int[] a, int[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (int i = 0; i < a.Length; i++)
        {
            // -1 means dynamic dimension, always matches
            if (a[i] == -1 || b[i] == -1)
            {
                continue;
            }

            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }
}
