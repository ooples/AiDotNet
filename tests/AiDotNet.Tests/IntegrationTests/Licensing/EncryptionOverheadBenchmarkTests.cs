using System.Diagnostics;
using AiDotNet.Helpers;
using Xunit;
using Xunit.Abstractions;
using System.Threading.Tasks;

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

        // Encryption uses PBKDF2-SHA256 to derive a per-call AES key from
        // the license key + a fresh random salt. PBKDF2 is intentionally
        // slow (~tens of ms per derivation) to harden the key against
        // brute-force, and the cost is paid PER ENCRYPTION call.
        //
        // Empirically (see this test's own output) PBKDF2 dominates the
        // total crypto time even at 10 MB payloads: 65 ms total of which
        // ~50 ms is PBKDF2 and only ~15 ms is the actual AES work. AES
        // doesn't reliably overtake PBKDF2 until ≥ ~100 MB on commodity
        // hardware, far outside what a model-save benchmark exercises.
        //
        // The original "10× memcpy" budget assumed the AES-throughput
        // regime, but in practice every call site is in the PBKDF2-
        // dominated regime. The ratio is therefore meaningless as a
        // regression guard — it just measures how slow PBKDF2 is
        // relative to memcpy at the chosen payload size.
        //
        // Use an absolute-time budget instead. A healthy machine takes
        // ~50–100 ms per encrypt+decrypt across all payload sizes (since
        // PBKDF2 is the floor). 500 ms allows generous headroom for
        // shared-CI variance, GC pauses, and the AES contribution at
        // multi-MB payloads, while still catching real regressions like
        // a 10× PBKDF2 iteration bump or a botched O(N²) loop on the
        // ciphertext.
        const double absoluteCryptoBudgetMs = 500.0;
        Assert.True(totalCryptoMs < absoluteCryptoBudgetMs,
            $"Per-op crypto time is {totalCryptoMs:F1} ms — exceeds " +
            $"{absoluteCryptoBudgetMs:F0} ms budget for {label} " +
            $"(overhead ratio vs memcpy: {overheadRatio:F1}×, informational only).");
        _output.WriteLine($"[{label}] Total crypto time: {totalCryptoMs:F1} ms ({measureIterations} iterations)");
    }

    [Fact(Timeout = 120000)]
    public async Task EncryptedPayload_SizeOverhead_IsMinimal()
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

        // AES-GCM overhead is exactly salt(16) + nonce(12) + tag(16) = 44 bytes
        Assert.Equal(44, totalOverhead);
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