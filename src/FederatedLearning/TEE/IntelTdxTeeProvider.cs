using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// TEE provider for Intel TDX (Trust Domain Extensions) confidential VMs.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Intel TDX is the next generation of Intel's confidential computing,
/// offering VM-level isolation instead of SGX's process-level enclaves. TDX protects an entire
/// virtual machine — the hypervisor cannot read or modify the VM's memory. This enables GB-scale
/// protected workloads, making it ideal for federated learning aggregation with large models.</para>
///
/// <para><b>TDX vs SGX:</b></para>
/// <list type="bullet">
/// <item><description><b>Memory:</b> TDX supports GB-scale (vs. SGX's 256 MB EPC).</description></item>
/// <item><description><b>Granularity:</b> TDX protects entire VM; SGX protects individual enclaves.</description></item>
/// <item><description><b>Compatibility:</b> TDX runs unmodified applications; SGX requires SGX SDK.</description></item>
/// <item><description><b>Attestation:</b> TDX uses TD Quote (ECDSA-based, similar to DCAP).</description></item>
/// </list>
///
/// <para><b>Recommended for FL:</b> TDX is the recommended TEE for federated learning because it
/// supports large model aggregation without the memory constraints of SGX.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class IntelTdxTeeProvider<T> : TeeProviderBase<T>
{
    /// <summary>
    /// TDX default max memory: 16 GB (configurable per TD).
    /// </summary>
    private const long TdxDefaultMaxBytes = 16L * 1024 * 1024 * 1024;

    private byte[] _tdReportData = Array.Empty<byte>();
    private byte[] _mrtd = Array.Empty<byte>(); // TD measurement (analogous to MRENCLAVE)

    /// <inheritdoc/>
    public override TeeProviderType ProviderType => TeeProviderType.Tdx;

    /// <inheritdoc/>
    public override void Initialize(TeeOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        // Generate MRTD (Trust Domain measurement)
        _mrtd = new byte[48]; // TDX uses 384-bit measurements
        _tdReportData = new byte[64];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(_mrtd);
            rng.GetBytes(_tdReportData);
        }

        base.Initialize(options);
    }

    /// <inheritdoc/>
    public override void Destroy()
    {
        if (_mrtd.Length > 0) Array.Clear(_mrtd, 0, _mrtd.Length);
        if (_tdReportData.Length > 0) Array.Clear(_tdReportData, 0, _tdReportData.Length);

        _mrtd = Array.Empty<byte>();
        _tdReportData = Array.Empty<byte>();

        base.Destroy();
    }

    /// <inheritdoc/>
    public override long GetMaxEnclaveMemory()
    {
        long requested = (long)Options.MaxEnclaveMemoryMb * 1024 * 1024;
        return Math.Min(requested, TdxDefaultMaxBytes);
    }

    /// <inheritdoc/>
    protected override byte[] DeriveSealingKey()
    {
        // TDX sealing key derived from MRTD
        var salt = System.Text.Encoding.UTF8.GetBytes("IntelTDX-SealingKey-MRTD-v1");
        var info = System.Text.Encoding.UTF8.GetBytes("AES-256-GCM-Sealing");
        return HkdfSha256.DeriveKey(_mrtd, salt, info, 32);
    }

    /// <inheritdoc/>
    protected override string ComputeMeasurementHash()
    {
        // TDX uses 384-bit MRTD
        return BitConverter.ToString(_mrtd).Replace("-", "").ToLowerInvariant();
    }

    /// <inheritdoc/>
    protected override byte[] BuildQuote(byte[] reportData)
    {
        // TDX TD Quote format (simplified):
        // version(2) + providerType(1) + timestamp(8) + MRTD(48) + reportDataLen(4) + reportData
        //   + ECDSA-signature(64)
        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        int quoteLength = 2 + 1 + 8 + 48 + 4 + reportData.Length + 64;
        var quote = new byte[quoteLength];
        int offset = 0;

        // Version (TDX Quote v4)
        quote[offset++] = 0x04;
        quote[offset++] = 0x00;

        // Provider type
        quote[offset++] = (byte)TeeProviderType.Tdx;

        // Timestamp
        var timestampBytes = BitConverter.GetBytes(timestamp);
        Buffer.BlockCopy(timestampBytes, 0, quote, offset, 8);
        offset += 8;

        // MRTD (48 bytes)
        Buffer.BlockCopy(_mrtd, 0, quote, offset, 48);
        offset += 48;

        // Report data
        var reportLenBytes = BitConverter.GetBytes(reportData.Length);
        Buffer.BlockCopy(reportLenBytes, 0, quote, offset, 4);
        offset += 4;
        Buffer.BlockCopy(reportData, 0, quote, offset, reportData.Length);
        offset += reportData.Length;

        // Simulated ECDSA signature
        using var hmac = new HMACSHA256(_mrtd);
        byte[] sig1 = hmac.ComputeHash(quote, 0, offset);
        Buffer.BlockCopy(sig1, 0, quote, offset, 32);
        offset += 32;

        using var hmac2 = new HMACSHA256(_tdReportData);
        byte[] sig2 = hmac2.ComputeHash(quote, 0, offset - 32);
        Buffer.BlockCopy(sig2, 0, quote, offset, 32);

        return quote;
    }
}
