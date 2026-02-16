namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for zero-knowledge verification in federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how client updates are verified before
/// aggregation. The default uses hash commitments with norm bound checking, which provides
/// a good balance of security and performance.</para>
/// </remarks>
public class VerificationOptions
{
    /// <summary>
    /// Gets or sets the ZK proof system to use. Default is <see cref="ZkProofSystem.Pedersen"/>.
    /// </summary>
    public ZkProofSystem ProofSystem { get; set; } = ZkProofSystem.Pedersen;

    /// <summary>
    /// Gets or sets the verification level. Default is <see cref="VerificationLevel.NormBound"/>.
    /// </summary>
    public VerificationLevel Level { get; set; } = VerificationLevel.NormBound;

    /// <summary>
    /// Gets or sets the gradient L2 norm bound for NormBound verification.
    /// Clients must prove ||g|| &lt;= this value. Default is 10.0.
    /// </summary>
    public double GradientNormBound { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the per-element bound for ElementBound verification.
    /// Each gradient component must be in [-B, B]. Default is 1.0.
    /// </summary>
    public double ElementBound { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the loss threshold for LossThreshold verification.
    /// Clients must prove their local loss is below this value. Default is 10.0.
    /// </summary>
    public double LossThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the security parameter in bits. Default is 128.
    /// </summary>
    public int SecurityParameterBits { get; set; } = 128;

    /// <summary>
    /// Gets or sets the maximum time allowed for proof generation per client, in milliseconds.
    /// Default is 30000 (30 seconds).
    /// </summary>
    public int ProofTimeoutMs { get; set; } = 30000;

    /// <summary>
    /// Gets or sets whether to reject clients that fail verification. Default is true.
    /// If false, failed clients are logged but still included in aggregation.
    /// </summary>
    public bool RejectFailedClients { get; set; } = true;

    /// <summary>
    /// Gets or sets the commitment options for the underlying commitment scheme.
    /// </summary>
    public CommitmentOptions Commitment { get; set; } = new CommitmentOptions();
}
