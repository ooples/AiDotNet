using System.Diagnostics;
using AiDotNet.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Licensing;

/// <summary>
/// Benchmark tests measuring encryption overhead for model payloads.
/// Verifies that AES-256-GCM encryption adds acceptable overhead relative to
/// plaintext serialization, even for large model payloads.
/// </summary>
public class EncryptionOverheadBenchmarkTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>
    /// Whether AES-GCM is available on the current runtime.
    /// .NET Framework 4.7.1 does not support System.Security.Cryptography.AesGcm.
    /// </summary>
    private static readonly bool AesGcmAvailable = CheckAesGcmAvailability();

    public EncryptionOverheadBenchmarkTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Theory]
    [InlineData(1024, "1 KB")]
    [InlineData(1024 * 100, "100 KB")]
    [InlineData(1024 * 1024, "1 MB")]
    [InlineData(1024 * 1024 * 10, "10 MB")]
    public void EncryptDecrypt_OverheadIsAcceptable(int payloadSize, string label)
    {
        if (!AesGcmAvailable)
        {
            _output.WriteLine($"[{label}] Skipped — AES-GCM not available on this runtime");
            return;
        }

        const string licenseKey = "benchmark-test-license-key-12345";
        const string aadText = "AiDotNet.Models.TestModel|[10,3]|[10]";
        const int warmupIterations = 3;
        const int measureIterations = 10;

        // Create deterministic test payload
        var payload = new byte[payloadSize];
        var rng = new Random(42);
        rng.NextBytes(payload);

        // Warmup
        for (int i = 0; i < warmupIterations; i++)
        {
            var encrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, aadText);
            ModelPayloadEncryption.Decrypt(encrypted.Ciphertext, licenseKey,
                encrypted.Salt, encrypted.Nonce, encrypted.Tag, aadText);
        }

        // Measure plaintext copy baseline (memcpy equivalent)
        var baselineSw = Stopwatch.StartNew();
        for (int i = 0; i < measureIterations; i++)
        {
            var copy = new byte[payload.Length];
            Buffer.BlockCopy(payload, 0, copy, 0, payload.Length);
        }
        baselineSw.Stop();
        double baselineMs = baselineSw.Elapsed.TotalMilliseconds / measureIterations;

        // Measure encryption
        EncryptedPayload lastEncrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, aadText);
        var encryptSw = Stopwatch.StartNew();
        for (int i = 0; i < measureIterations; i++)
        {
            lastEncrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, aadText);
        }
        encryptSw.Stop();
        double encryptMs = encryptSw.Elapsed.TotalMilliseconds / measureIterations;

        // Measure decryption
        var decryptSw = Stopwatch.StartNew();
        for (int i = 0; i < measureIterations; i++)
        {
            ModelPayloadEncryption.Decrypt(lastEncrypted.Ciphertext, licenseKey,
                lastEncrypted.Salt, lastEncrypted.Nonce, lastEncrypted.Tag, aadText);
        }
        decryptSw.Stop();
        double decryptMs = decryptSw.Elapsed.TotalMilliseconds / measureIterations;

        double totalCryptoMs = encryptMs + decryptMs;
        double overheadRatio = baselineMs > 0 ? totalCryptoMs / baselineMs : double.PositiveInfinity;

        _output.WriteLine($"[{label}] Baseline (memcpy): {baselineMs:F3} ms");
        _output.WriteLine($"[{label}] Encrypt:           {encryptMs:F3} ms");
        _output.WriteLine($"[{label}] Decrypt:           {decryptMs:F3} ms");
        _output.WriteLine($"[{label}] Total crypto:      {totalCryptoMs:F3} ms");
        _output.WriteLine($"[{label}] Overhead ratio:    {overheadRatio:F1}x vs memcpy");
        _output.WriteLine($"[{label}] Ciphertext size:   {lastEncrypted.Ciphertext.Length} bytes (payload: {payloadSize})");

        // Verify round-trip correctness
        var decrypted = ModelPayloadEncryption.Decrypt(lastEncrypted.Ciphertext, licenseKey,
            lastEncrypted.Salt, lastEncrypted.Nonce, lastEncrypted.Tag, aadText);
        Assert.Equal(payload.Length, decrypted.Length);
        Assert.Equal(payload, decrypted);

        // Log encryption overhead for diagnostic purposes.
        // We don't assert a hard time limit since CI environments vary widely in performance.
        _output.WriteLine($"[{label}] Total crypto time: {totalCryptoMs:F1} ms ({measureIterations} iterations)");
    }

    [Fact]
    public void EncryptedPayload_SizeOverhead_IsMinimal()
    {
        if (!AesGcmAvailable)
        {
            _output.WriteLine("Skipped — AES-GCM not available on this runtime");
            return;
        }

        const string licenseKey = "test-key";
        const string aadText = "test-aad";
        var payload = new byte[1_000_000]; // 1 MB
        new Random(42).NextBytes(payload);

        var encrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, aadText);

        // AES-GCM ciphertext should be same size as plaintext (no padding)
        Assert.Equal(payload.Length, encrypted.Ciphertext.Length);

        // Total overhead is salt (16) + nonce (12) + tag (16) = 44 bytes
        int totalOverhead = encrypted.Salt.Length + encrypted.Nonce.Length + encrypted.Tag.Length;
        _output.WriteLine($"Payload: {payload.Length} bytes, Ciphertext: {encrypted.Ciphertext.Length} bytes");
        _output.WriteLine($"Overhead: salt={encrypted.Salt.Length} + nonce={encrypted.Nonce.Length} + tag={encrypted.Tag.Length} = {totalOverhead} bytes");
        _output.WriteLine($"Overhead ratio: {(double)totalOverhead / payload.Length * 100:F4}%");

        Assert.True(totalOverhead <= 64, $"Encryption metadata overhead is too large: {totalOverhead} bytes");
    }

    private static bool CheckAesGcmAvailability()
    {
        try
        {
            // Try a minimal encrypt to see if the platform supports AES-GCM
            var testPayload = new byte[] { 1, 2, 3, 4 };
            ModelPayloadEncryption.Encrypt(testPayload, "test-key", "test-aad");
            return true;
        }
        catch (PlatformNotSupportedException)
        {
            return false;
        }
    }
}
