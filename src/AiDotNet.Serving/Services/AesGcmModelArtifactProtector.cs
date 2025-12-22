using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Protects model artifacts using AES-256-GCM.
/// </summary>
public sealed class AesGcmModelArtifactProtector : IModelArtifactProtector
{
    private const int NonceSize = 12;
    private const int TagSize = 16;
    private static readonly byte[] Magic = "AIDN"u8.ToArray();

    public ProtectedModelArtifact ProtectToFile(string modelName, string sourcePath, string outputDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name is required.", nameof(modelName));
        }

        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            throw new ArgumentException("Source path is required.", nameof(sourcePath));
        }

        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Output directory is required.", nameof(outputDirectory));
        }

        Directory.CreateDirectory(outputDirectory);

        var plaintext = File.ReadAllBytes(sourcePath);
        var key = RandomNumberGenerator.GetBytes(32);
        var nonce = RandomNumberGenerator.GetBytes(NonceSize);
        var tag = new byte[TagSize];
        var ciphertext = new byte[plaintext.Length];

        var aad = Encoding.UTF8.GetBytes(modelName);

        try
        {
            using (var aes = new AesGcm(key, TagSize))
            {
                aes.Encrypt(nonce, plaintext, ciphertext, tag, aad);
            }

            var outputPath = Path.Combine(outputDirectory, $"{SanitizeFileName(modelName)}.aidn.enc");
            using (var stream = File.Open(outputPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                stream.Write(Magic, 0, Magic.Length);
                stream.WriteByte(1); // version
                stream.Write(nonce, 0, nonce.Length);
                stream.Write(tag, 0, tag.Length);
                stream.Write(ciphertext, 0, ciphertext.Length);
            }

            var keyId = Guid.NewGuid().ToString("N");
            return new ProtectedModelArtifact(modelName, outputPath, keyId, key, nonce, "AES-256-GCM");
        }
        finally
        {
            CryptographicOperations.ZeroMemory(plaintext);
            CryptographicOperations.ZeroMemory(ciphertext);
            CryptographicOperations.ZeroMemory(tag);
            CryptographicOperations.ZeroMemory(aad);

            // ProtectedModelArtifact makes defensive copies, so it is safe to clear local buffers.
            CryptographicOperations.ZeroMemory(key);
            CryptographicOperations.ZeroMemory(nonce);
        }
    }

    private static string SanitizeFileName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = name.Select(c => invalid.Contains(c) ? '_' : c).ToArray();
        return new string(chars);
    }
}

