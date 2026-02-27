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
/// proxy verifies them using lightweight checks, and issues a "verified" certificate to the
/// server. This reduces the ZK proof overhead while maintaining verifiability.</para>
///
/// <para>Protocol:</para>
/// <list type="number">
/// <item>Client computes update and commitment: C = SHA256(update || nonce)</item>
/// <item>Client sends (update, C, nonce) to proxy</item>
/// <item>Proxy verifies: recomputes hash, checks norm bounds, element bounds</item>
/// <item>Proxy sends (update, certificate) to server (proxy cannot modify update)</item>
/// </list>
///
/// <para>Reference: Proxy-based ZKP for Efficient Federated Verification (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class ProxyZKPVerifier<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _maxNorm;
    private readonly double _maxElementMagnitude;

    /// <summary>
    /// Creates a new Proxy ZKP verifier.
    /// </summary>
    /// <param name="maxNorm">Maximum allowed L2 norm for updates. Default: 10.0.</param>
    /// <param name="maxElementMagnitude">Maximum allowed magnitude per element. Default: 1.0.</param>
    public ProxyZKPVerifier(double maxNorm = 10.0, double maxElementMagnitude = 1.0)
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
    }

    /// <summary>
    /// Computes a commitment hash for a client update: C = SHA256(update_bytes || nonce).
    /// Clients call this before sending their update to create a binding commitment.
    /// </summary>
    /// <param name="update">The model update to commit to.</param>
    /// <param name="nonce">A random nonce for binding (prevents replay attacks).</param>
    /// <returns>Hex-encoded SHA-256 commitment hash.</returns>
    public string ComputeCommitment(Dictionary<string, T[]> update, byte[] nonce)
    {
        byte[] updateBytes = SerializeUpdate(update);

        using var sha256 = SHA256.Create();
        // Hash(update || nonce)
        sha256.TransformBlock(updateBytes, 0, updateBytes.Length, null, 0);
        sha256.TransformFinalBlock(nonce, 0, nonce.Length);

        byte[] hash = sha256.Hash!;
        return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
    }

    /// <summary>
    /// Verifies a client update against proxy constraints including commitment hash verification.
    /// </summary>
    /// <param name="update">Client's model update.</param>
    /// <param name="commitment">Client's commitment hash (hex string from ComputeCommitment).</param>
    /// <param name="nonce">The nonce used when computing the commitment.</param>
    /// <returns>Verification result with pass/fail and reason.</returns>
    public ProxyVerificationResult Verify(Dictionary<string, T[]> update, string commitment, byte[] nonce)
    {
        // Check 1: Commitment must be present.
        if (string.IsNullOrEmpty(commitment))
        {
            return new ProxyVerificationResult(false, "Missing commitment hash.");
        }

        if (nonce == null || nonce.Length == 0)
        {
            return new ProxyVerificationResult(false, "Missing nonce.");
        }

        // Check 2: Verify commitment hash matches C = SHA256(update || nonce).
        string recomputedCommitment = ComputeCommitment(update, nonce);
        if (!string.Equals(commitment, recomputedCommitment, StringComparison.OrdinalIgnoreCase))
        {
            return new ProxyVerificationResult(false,
                "Commitment hash mismatch: update was modified after commitment.");
        }

        // Check 3: Per-element magnitude bounds.
        double totalNorm2 = 0;
        foreach (var kvp in update)
        {
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                double v = NumOps.ToDouble(kvp.Value[i]);

                if (double.IsNaN(v) || double.IsInfinity(v))
                {
                    return new ProxyVerificationResult(false,
                        $"NaN/Infinity detected in layer '{kvp.Key}' at index {i}.");
                }

                if (Math.Abs(v) > _maxElementMagnitude)
                {
                    return new ProxyVerificationResult(false,
                        $"Element magnitude {Math.Abs(v):F4} exceeds limit {_maxElementMagnitude} in layer '{kvp.Key}'.");
                }

                totalNorm2 += v * v;
            }
        }

        // Check 4: L2 norm bound.
        double norm = Math.Sqrt(totalNorm2);
        if (norm > _maxNorm)
        {
            return new ProxyVerificationResult(false,
                $"Update norm {norm:F4} exceeds limit {_maxNorm}.");
        }

        return new ProxyVerificationResult(true, "Verified.", recomputedCommitment, norm);
    }

    /// <summary>
    /// Overload for backward compatibility — verifies without nonce (commitment-only check skipped
    /// if nonce is unavailable, but norm/element checks still apply).
    /// </summary>
    public ProxyVerificationResult Verify(Dictionary<string, T[]> update, string commitment)
    {
        if (string.IsNullOrEmpty(commitment))
        {
            return new ProxyVerificationResult(false, "Missing commitment hash.");
        }

        double totalNorm2 = 0;
        foreach (var kvp in update)
        {
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                double v = NumOps.ToDouble(kvp.Value[i]);

                if (double.IsNaN(v) || double.IsInfinity(v))
                {
                    return new ProxyVerificationResult(false,
                        $"NaN/Infinity detected in layer '{kvp.Key}' at index {i}.");
                }

                if (Math.Abs(v) > _maxElementMagnitude)
                {
                    return new ProxyVerificationResult(false,
                        $"Element magnitude {Math.Abs(v):F4} exceeds limit {_maxElementMagnitude} in layer '{kvp.Key}'.");
                }

                totalNorm2 += v * v;
            }
        }

        double norm = Math.Sqrt(totalNorm2);
        if (norm > _maxNorm)
        {
            return new ProxyVerificationResult(false,
                $"Update norm {norm:F4} exceeds limit {_maxNorm}.");
        }

        return new ProxyVerificationResult(true, "Verified (commitment not re-verified without nonce).", commitment, norm);
    }

    /// <summary>
    /// Generates a cryptographically secure random nonce.
    /// </summary>
    /// <param name="length">Nonce length in bytes. Default: 32.</param>
    /// <returns>Random nonce bytes.</returns>
    public static byte[] GenerateNonce(int length = 32)
    {
        var nonce = new byte[length];
        using var rng = RandomNumberGenerator.Create();
        rng.GetBytes(nonce);
        return nonce;
    }

    private byte[] SerializeUpdate(Dictionary<string, T[]> update)
    {
        // Deterministic serialization: sorted layer names, then IEEE 754 doubles.
        var sortedKeys = update.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
        int totalParams = 0;
        foreach (var key in sortedKeys)
        {
            totalParams += update[key].Length;
        }

        // Layer name bytes + param doubles.
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
/// Result of a proxy ZKP verification.
/// </summary>
public class ProxyVerificationResult
{
    /// <summary>Creates a new verification result.</summary>
    public ProxyVerificationResult(bool isValid, string reason, string? commitmentHash = null, double? updateNorm = null)
    {
        IsValid = isValid;
        Reason = reason;
        CommitmentHash = commitmentHash;
        UpdateNorm = updateNorm;
    }

    /// <summary>Whether the verification passed.</summary>
    public bool IsValid { get; }

    /// <summary>Human-readable reason for the result.</summary>
    public string Reason { get; }

    /// <summary>The verified commitment hash, if available.</summary>
    public string? CommitmentHash { get; }

    /// <summary>The L2 norm of the update, if computed.</summary>
    public double? UpdateNorm { get; }
}
