using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Implements Proxy-based Zero-Knowledge Proof verification for federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard ZK proofs can be expensive for clients to generate.
/// ProxyZKP uses a semi-trusted proxy (e.g., a TEE enclave or an auditor) that verifies client
/// computations without seeing the raw data. The client sends commitments to the proxy, the
/// proxy verifies them using lightweight checks, and issues a cryptographically-signed certificate
/// to the server. This reduces the ZK proof overhead while maintaining verifiability.</para>
///
/// <para>Protocol:</para>
/// <list type="number">
/// <item>Client computes update and commitment: C = HMAC-SHA256(proxyKey, update_bytes || nonce)</item>
/// <item>Client sends (update, C, nonce) to proxy</item>
/// <item>Proxy verifies: recomputes HMAC, checks norm bounds, element bounds</item>
/// <item>If verified, proxy issues a signed certificate: Cert = HMAC-SHA256(proxyKey, commitment || status || timestamp)</item>
/// <item>Server receives (update, certificate) and verifies the certificate signature</item>
/// </list>
///
/// <para>Reference: Proxy-based ZKP for Efficient Federated Verification (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class ProxyZKPVerifier<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _maxNorm;
    private readonly double _maxElementMagnitude;
    private readonly byte[] _proxyKey;
    private readonly TimeSpan _certificateValidity;

    /// <summary>
    /// Creates a new Proxy ZKP verifier.
    /// </summary>
    /// <param name="maxNorm">Maximum allowed L2 norm for updates. Default: 10.0.</param>
    /// <param name="maxElementMagnitude">Maximum allowed magnitude per element. Default: 1.0.</param>
    /// <param name="proxyKey">HMAC signing key for the proxy. If null, a random 256-bit key is generated.</param>
    /// <param name="certificateValidityMinutes">How long a certificate is valid. Default: 30 minutes.</param>
    public ProxyZKPVerifier(
        double maxNorm = 10.0,
        double maxElementMagnitude = 1.0,
        byte[]? proxyKey = null,
        int certificateValidityMinutes = 30)
    {
        if (maxNorm <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxNorm), "Max norm must be positive.");
        }

        if (maxElementMagnitude <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxElementMagnitude), "Max element magnitude must be positive.");
        }

        _maxNorm = maxNorm;
        _maxElementMagnitude = maxElementMagnitude;
        _certificateValidity = TimeSpan.FromMinutes(certificateValidityMinutes);

        if (proxyKey != null && proxyKey.Length > 0)
        {
            _proxyKey = (byte[])proxyKey.Clone();
        }
        else
        {
            _proxyKey = new byte[32];
            using var rng = RandomNumberGenerator.Create();
            rng.GetBytes(_proxyKey);
        }
    }

    /// <summary>
    /// Computes an HMAC-SHA256 commitment for a client update.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> HMAC (Hash-based Message Authentication Code) produces a tag
    /// that binds the update to a specific nonce and the proxy's secret key. Unlike plain SHA256,
    /// HMAC prevents an attacker from forging a commitment without knowledge of the key. The nonce
    /// ensures each commitment is unique even for identical updates.</para>
    /// </remarks>
    /// <param name="update">The model update to commit to.</param>
    /// <param name="nonce">A random nonce for binding (prevents replay attacks).</param>
    /// <returns>Hex-encoded HMAC-SHA256 commitment.</returns>
    public string ComputeCommitment(Dictionary<string, T[]> update, byte[] nonce)
    {
        Guard.NotNull(update);
        Guard.NotNull(nonce);
        byte[] updateBytes = SerializeUpdate(update);

        using var hmac = new HMACSHA256(_proxyKey);
        var combined = new byte[updateBytes.Length + nonce.Length];
        Buffer.BlockCopy(updateBytes, 0, combined, 0, updateBytes.Length);
        Buffer.BlockCopy(nonce, 0, combined, updateBytes.Length, nonce.Length);

        byte[] tag = hmac.ComputeHash(combined);
        return BitConverter.ToString(tag).Replace("-", "").ToLowerInvariant();
    }

    /// <summary>
    /// Verifies a client update and issues a signed certificate if all checks pass.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The proxy performs three types of checks:</para>
    /// <list type="bullet">
    /// <item><b>Integrity:</b> The commitment hash matches — the update wasn't tampered with.</item>
    /// <item><b>Element bounds:</b> No individual parameter is suspiciously large (potential attack).</item>
    /// <item><b>Norm bound:</b> The overall update magnitude is within expected range.</item>
    /// </list>
    /// <para>If all checks pass, the proxy signs a certificate that the server can verify later
    /// without trusting the client.</para>
    /// </remarks>
    /// <param name="update">Client's model update.</param>
    /// <param name="commitment">Client's commitment hash from <see cref="ComputeCommitment"/>.</param>
    /// <param name="nonce">The nonce used when computing the commitment.</param>
    /// <returns>Verification result with certificate if successful.</returns>
    public ProxyVerificationResult Verify(Dictionary<string, T[]> update, string commitment, byte[] nonce)
    {
        Guard.NotNull(update);
        if (string.IsNullOrEmpty(commitment))
        {
            return ProxyVerificationResult.Fail("Missing commitment hash.");
        }

        if (nonce == null || nonce.Length == 0)
        {
            return ProxyVerificationResult.Fail("Missing nonce.");
        }

        // Check 1: Verify HMAC commitment — ensures update integrity.
        string recomputedCommitment = ComputeCommitment(update, nonce);
        if (!CryptographicEquals(commitment, recomputedCommitment))
        {
            return ProxyVerificationResult.Fail(
                "Commitment hash mismatch: update was modified after commitment.");
        }

        // Check 2: Per-element magnitude bounds and NaN/Infinity.
        double totalNorm2 = 0;
        foreach (var kvp in update)
        {
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                double v = NumOps.ToDouble(kvp.Value[i]);

                if (double.IsNaN(v) || double.IsInfinity(v))
                {
                    return ProxyVerificationResult.Fail(
                        $"NaN/Infinity detected in layer '{kvp.Key}' at index {i}.");
                }

                if (Math.Abs(v) > _maxElementMagnitude)
                {
                    return ProxyVerificationResult.Fail(
                        $"Element magnitude {Math.Abs(v):F4} exceeds limit {_maxElementMagnitude} in layer '{kvp.Key}'.");
                }

                totalNorm2 += v * v;
            }
        }

        // Check 3: L2 norm bound.
        double norm = Math.Sqrt(totalNorm2);
        if (norm > _maxNorm)
        {
            return ProxyVerificationResult.Fail(
                $"Update norm {norm:F4} exceeds limit {_maxNorm}.");
        }

        // All checks passed — issue signed certificate.
        var certificate = IssueCertificate(recomputedCommitment, norm);
        return ProxyVerificationResult.Pass(recomputedCommitment, norm, certificate);
    }

    /// <summary>
    /// Issues a signed proxy certificate attesting that the update passed verification.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A certificate is like a signed letter from the proxy saying
    /// "I checked this update and it's legitimate." The server can verify the signature to confirm
    /// the proxy actually issued it, without having to re-run all the checks itself.</para>
    /// </remarks>
    /// <param name="commitment">The verified commitment hash.</param>
    /// <param name="updateNorm">The computed L2 norm of the update.</param>
    /// <returns>A signed certificate.</returns>
    public ProxyCertificate IssueCertificate(string commitment, double updateNorm)
    {
        long issuedAtTicks = DateTime.UtcNow.Ticks;
        long expiresAtTicks = issuedAtTicks + _certificateValidity.Ticks;

        // Build the message to sign: commitment + status + norm + timestamps.
        string signatureInput = $"{commitment}|VERIFIED|{updateNorm:R}|{issuedAtTicks}|{expiresAtTicks}";
        byte[] inputBytes = Encoding.UTF8.GetBytes(signatureInput);

        using var hmac = new HMACSHA256(_proxyKey);
        byte[] signatureBytes = hmac.ComputeHash(inputBytes);
        string signature = BitConverter.ToString(signatureBytes).Replace("-", "").ToLowerInvariant();

        return new ProxyCertificate(
            commitmentHash: commitment,
            updateNorm: updateNorm,
            issuedAtTicks: issuedAtTicks,
            expiresAtTicks: expiresAtTicks,
            signature: signature);
    }

    /// <summary>
    /// Verifies a proxy certificate's signature and expiry. The server calls this to confirm
    /// the proxy actually issued the certificate and it hasn't expired.
    /// </summary>
    /// <param name="certificate">The certificate to verify.</param>
    /// <returns>True if the certificate is valid and not expired.</returns>
    public bool VerifyCertificate(ProxyCertificate certificate)
    {
        if (certificate == null)
        {
            return false;
        }

        // Check expiry.
        if (DateTime.UtcNow.Ticks > certificate.ExpiresAtTicks)
        {
            return false;
        }

        // Recompute the expected signature.
        string signatureInput = $"{certificate.CommitmentHash}|VERIFIED|{certificate.UpdateNorm:R}|{certificate.IssuedAtTicks}|{certificate.ExpiresAtTicks}";
        byte[] inputBytes = Encoding.UTF8.GetBytes(signatureInput);

        using var hmac = new HMACSHA256(_proxyKey);
        byte[] expectedBytes = hmac.ComputeHash(inputBytes);
        string expectedSignature = BitConverter.ToString(expectedBytes).Replace("-", "").ToLowerInvariant();

        return CryptographicEquals(certificate.Signature, expectedSignature);
    }

    /// <summary>
    /// Generates a cryptographically secure random nonce.
    /// </summary>
    /// <param name="length">Nonce length in bytes. Default: 32.</param>
    /// <returns>Random nonce bytes.</returns>
    public static byte[] GenerateNonce(int length = 32)
    {
        if (length < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(length), "Nonce length must be at least 1.");
        }

        var nonce = new byte[length];
        using var rng = RandomNumberGenerator.Create();
        rng.GetBytes(nonce);
        return nonce;
    }

    /// <summary>
    /// Constant-time string comparison to prevent timing attacks on hash/signature comparison.
    /// </summary>
    private static bool CryptographicEquals(string a, string b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        int diff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            diff |= a[i] ^ b[i];
        }

        return diff == 0;
    }

    private byte[] SerializeUpdate(Dictionary<string, T[]> update)
    {
        // Deterministic serialization: sorted layer names, then IEEE 754 doubles.
        var sortedKeys = update.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();

        using var ms = new System.IO.MemoryStream();
        using var writer = new System.IO.BinaryWriter(ms, Encoding.UTF8);

        writer.Write(sortedKeys.Count);
        foreach (var key in sortedKeys)
        {
            writer.Write(key);
            var values = update[key];
            writer.Write(values.Length);
            for (int i = 0; i < values.Length; i++)
            {
                writer.Write(NumOps.ToDouble(values[i]));
            }
        }

        writer.Flush();
        return ms.ToArray();
    }

    /// <summary>Gets the maximum allowed norm.</summary>
    public double MaxNorm => _maxNorm;

    /// <summary>Gets the maximum allowed element magnitude.</summary>
    public double MaxElementMagnitude => _maxElementMagnitude;
}

/// <summary>
/// A signed certificate issued by the proxy after verifying a client update.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is like a stamp of approval from the proxy. It contains
/// what was verified (the commitment hash and norm), when (timestamps), and a cryptographic
/// signature that proves the proxy issued it. The server can check this signature to trust the
/// update without re-running all checks.</para>
/// </remarks>
public class ProxyCertificate
{
    /// <summary>Creates a new proxy certificate.</summary>
    public ProxyCertificate(string commitmentHash, double updateNorm, long issuedAtTicks, long expiresAtTicks, string signature)
    {
        CommitmentHash = commitmentHash;
        UpdateNorm = updateNorm;
        IssuedAtTicks = issuedAtTicks;
        ExpiresAtTicks = expiresAtTicks;
        Signature = signature;
    }

    /// <summary>The commitment hash that was verified.</summary>
    public string CommitmentHash { get; }

    /// <summary>The L2 norm of the verified update.</summary>
    public double UpdateNorm { get; }

    /// <summary>When the certificate was issued (UTC ticks).</summary>
    public long IssuedAtTicks { get; }

    /// <summary>When the certificate expires (UTC ticks).</summary>
    public long ExpiresAtTicks { get; }

    /// <summary>HMAC-SHA256 signature over the certificate contents.</summary>
    public string Signature { get; }

    /// <summary>Whether the certificate has expired.</summary>
    public bool IsExpired => DateTime.UtcNow.Ticks > ExpiresAtTicks;
}

/// <summary>
/// Result of a proxy ZKP verification.
/// </summary>
public class ProxyVerificationResult
{
    /// <summary>Creates a new verification result.</summary>
    public ProxyVerificationResult(bool isValid, string reason, string? commitmentHash = null, double? updateNorm = null, ProxyCertificate? certificate = null)
    {
        IsValid = isValid;
        Reason = reason;
        CommitmentHash = commitmentHash;
        UpdateNorm = updateNorm;
        Certificate = certificate;
    }

    /// <summary>Whether the verification passed.</summary>
    public bool IsValid { get; }

    /// <summary>Human-readable reason for the result.</summary>
    public string Reason { get; }

    /// <summary>The verified commitment hash, if available.</summary>
    public string? CommitmentHash { get; }

    /// <summary>The L2 norm of the update, if computed.</summary>
    public double? UpdateNorm { get; }

    /// <summary>Signed proxy certificate, available only if verification passed.</summary>
    public ProxyCertificate? Certificate { get; }

    /// <summary>Creates a failed verification result.</summary>
    public static ProxyVerificationResult Fail(string reason) =>
        new(false, reason);

    /// <summary>Creates a successful verification result with certificate.</summary>
    public static ProxyVerificationResult Pass(string commitmentHash, double updateNorm, ProxyCertificate certificate) =>
        new(true, "Verified.", commitmentHash, updateNorm, certificate);
}
