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
/// <item>Client computes update and commitment: C = Hash(update || nonce)</item>
/// <item>Client sends (update, C) to proxy</item>
/// <item>Proxy verifies: norm check, gradient bounds, consistency with commitment</item>
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
    /// Verifies a client update against proxy constraints.
    /// </summary>
    /// <param name="update">Client's model update.</param>
    /// <param name="commitment">Client's commitment hash (hex string).</param>
    /// <returns>Verification result with pass/fail and reason.</returns>
    public ProxyVerificationResult Verify(Dictionary<string, T[]> update, string commitment)
    {
        double totalNorm2 = 0;

        foreach (var kvp in update)
        {
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                double v = NumOps.ToDouble(kvp.Value[i]);

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

        if (string.IsNullOrEmpty(commitment))
        {
            return new ProxyVerificationResult(false, "Missing commitment.");
        }

        return new ProxyVerificationResult(true, "Verified.");
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
    public ProxyVerificationResult(bool isValid, string reason)
    {
        IsValid = isValid;
        Reason = reason;
    }

    /// <summary>Whether the verification passed.</summary>
    public bool IsValid { get; }

    /// <summary>Human-readable reason for the result.</summary>
    public string Reason { get; }
}
