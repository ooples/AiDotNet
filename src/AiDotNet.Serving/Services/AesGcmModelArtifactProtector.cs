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

    /// <summary>
    /// Cross-platform-invalid filename characters. Combines the Windows
    /// invalid set (most restrictive: ":" + "\\" + reserved punctuation +
    /// control chars) with POSIX "/" and "\0". Used instead of
    /// <see cref="Path.GetInvalidFileNameChars"/> because that method
    /// returns a platform-specific set — on Linux it only contains '\0'
    /// and '/', so a model name like "my:model" sanitizes to "my:model"
    /// on Linux but "my_model" on Windows. Encrypted artifacts are
    /// designed to be portable, so we apply the strict Windows superset
    /// on every OS to guarantee the output is mountable everywhere.
    /// </summary>
    private static readonly HashSet<char> CrossPlatformInvalidFileNameChars =
        new(new[]
        {
            '\0', '/', '\\', ':', '*', '?', '"', '<', '>', '|',
        }
        .Concat(Enumerable.Range(1, 31).Select(i => (char)i)));

    /// <summary>
    /// DOS reserved device names. Creating a file with any of these as the
    /// base name (with or without extension) fails on Windows with
    /// <c>PathTooLongException</c> / <c>IOException</c> because the kernel
    /// still routes them to legacy character devices. Cross-platform
    /// portability requires rejecting them even on POSIX hosts so an
    /// artifact produced on Linux can't be loaded on Windows.
    /// </summary>
    private static readonly HashSet<string> WindowsReservedFileNames =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        };

    private static string SanitizeFileName(string name)
    {
        // 1. Replace cross-platform-invalid characters.
        var chars = name.Select(c => CrossPlatformInvalidFileNameChars.Contains(c) ? '_' : c).ToArray();
        var sanitized = new string(chars);

        // 2. Windows strips trailing dots and spaces from filenames at create-time
        //    (so "model." silently becomes "model", but "model." on some paths fails
        //    with PathNotFound). Trim on every platform to avoid the mismatch.
        sanitized = sanitized.TrimEnd(' ', '.');

        // 3. If the base (pre-extension) is a reserved DOS device name, prefix it
        //    so the artifact remains portable. Split on the first dot so "NUL.bin"
        //    also gets rewritten.
        if (sanitized.Length == 0)
        {
            return "_";
        }

        var dotIndex = sanitized.IndexOf('.');
        var baseName = dotIndex >= 0 ? sanitized.Substring(0, dotIndex) : sanitized;
        if (WindowsReservedFileNames.Contains(baseName))
        {
            sanitized = "_" + sanitized;
        }

        return sanitized;
    }
}

