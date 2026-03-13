using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// TEE provider for Intel SGX (Software Guard Extensions) process-level enclaves.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Intel SGX creates small, secure "enclaves" inside a regular process.
/// Data inside an enclave is encrypted in memory and invisible to the OS, hypervisor, or other
/// processes. SGX enclaves are limited to ~256 MB of protected memory (EPC), making them suitable
/// for aggregation of model updates but not for training full models.</para>
///
/// <para><b>Key SGX concepts:</b></para>
/// <list type="bullet">
/// <item><description><b>MRENCLAVE:</b> Hash of the enclave code — changes if the code changes.</description></item>
/// <item><description><b>MRSIGNER:</b> Hash of the enclave author's signing key.</description></item>
/// <item><description><b>EPC:</b> Enclave Page Cache — the hardware-encrypted memory region (typically 128-256 MB).</description></item>
/// <item><description><b>DCAP:</b> Data Center Attestation Primitives — Intel's server-side attestation protocol.</description></item>
/// </list>
///
/// <para><b>Production use:</b> Requires an SGX-capable CPU (Xeon Scalable 3rd gen+) and the Intel
/// SGX SDK/PSW installed. This class models the SGX enclave lifecycle; actual SGX calls would be
/// made via P/Invoke to the SGX SDK in a production deployment.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class IntelSgxTeeProvider<T> : TeeProviderBase<T>
{
    /// <summary>
    /// SGX Enclave Page Cache limit: 256 MB.
    /// </summary>
    private const long SgxEpcLimitBytes = 256L * 1024 * 1024;

    private byte[] _mrEnclave = Array.Empty<byte>();
    private byte[] _mrSigner = Array.Empty<byte>();

    /// <inheritdoc/>
    public override TeeProviderType ProviderType => TeeProviderType.Sgx;

    /// <inheritdoc/>
    public override void Initialize(TeeOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        // Validate memory fits within SGX EPC
        long requestedBytes = (long)options.MaxEnclaveMemoryMb * 1024 * 1024;
        if (requestedBytes > SgxEpcLimitBytes)
        {
            throw new InvalidOperationException(
                $"Requested enclave memory ({options.MaxEnclaveMemoryMb} MB) exceeds SGX EPC limit (256 MB). " +
                "Consider using Intel TDX for larger memory requirements.");
        }

        // Generate MRENCLAVE and MRSIGNER identities
        _mrEnclave = new byte[32];
        _mrSigner = new byte[32];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(_mrEnclave);
            rng.GetBytes(_mrSigner);
        }

        base.Initialize(options);
    }

    /// <inheritdoc/>
    public override void Destroy()
    {
        if (_mrEnclave.Length > 0) Array.Clear(_mrEnclave, 0, _mrEnclave.Length);
        if (_mrSigner.Length > 0) Array.Clear(_mrSigner, 0, _mrSigner.Length);

        _mrEnclave = Array.Empty<byte>();
        _mrSigner = Array.Empty<byte>();

        base.Destroy();
    }

    /// <inheritdoc/>
    public override long GetMaxEnclaveMemory()
    {
        long requested = (long)Options.MaxEnclaveMemoryMb * 1024 * 1024;
        return Math.Min(requested, SgxEpcLimitBytes);
    }

    /// <inheritdoc/>
    protected override byte[] DeriveSealingKey()
    {
        // SGX sealing key is derived from MRENCLAVE (code identity)
        var salt = System.Text.Encoding.UTF8.GetBytes("IntelSGX-SealingKey-MRENCLAVE-v1");
        var info = System.Text.Encoding.UTF8.GetBytes("AES-256-GCM-Sealing");
        return HkdfSha256.DeriveKey(_mrEnclave, salt, info, 32);
    }

    /// <inheritdoc/>
    protected override string ComputeMeasurementHash()
    {
        return BitConverter.ToString(_mrEnclave).Replace("-", "").ToLowerInvariant();
    }

    /// <inheritdoc/>
    protected override byte[] BuildQuote(byte[] reportData)
    {
        // SGX DCAP quote format (simplified):
        // version(2) + providerType(1) + timestamp(8) + MRENCLAVE(32) + MRSIGNER(32)
        //   + reportDataLen(4) + reportData + ECDSA-signature(64)
        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        int quoteLength = 2 + 1 + 8 + 32 + 32 + 4 + reportData.Length + 64;
        var quote = new byte[quoteLength];
        int offset = 0;

        // Version (DCAP v3-style)
        quote[offset++] = 0x03;
        quote[offset++] = 0x00;

        // Provider type
        quote[offset++] = (byte)TeeProviderType.Sgx;

        // Timestamp
        var timestampBytes = BitConverter.GetBytes(timestamp);
        Buffer.BlockCopy(timestampBytes, 0, quote, offset, 8);
        offset += 8;

        // MRENCLAVE
        Buffer.BlockCopy(_mrEnclave, 0, quote, offset, 32);
        offset += 32;

        // MRSIGNER
        Buffer.BlockCopy(_mrSigner, 0, quote, offset, 32);
        offset += 32;

        // Report data length + data
        var reportLenBytes = BitConverter.GetBytes(reportData.Length);
        Buffer.BlockCopy(reportLenBytes, 0, quote, offset, 4);
        offset += 4;
        Buffer.BlockCopy(reportData, 0, quote, offset, reportData.Length);
        offset += reportData.Length;

        // Simulated ECDSA signature (HMAC-SHA256 doubled for 64 bytes)
        using var hmac = new HMACSHA256(_mrEnclave);
        byte[] sig = hmac.ComputeHash(quote, 0, offset);
        Buffer.BlockCopy(sig, 0, quote, offset, 32);
        offset += 32;
        // Second half from MRSIGNER-based HMAC
        using var hmac2 = new HMACSHA256(_mrSigner);
        byte[] sig2 = hmac2.ComputeHash(quote, 0, offset - 32);
        Buffer.BlockCopy(sig2, 0, quote, offset, 32);

        return quote;
    }
}
