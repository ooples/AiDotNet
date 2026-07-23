namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the level of zero-knowledge verification applied to client updates.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Higher verification levels provide stronger guarantees
/// but cost more computation time. Choose based on your threat model:</para>
/// <list type="bullet">
/// <item><description><b>None:</b> No verification. Trust clients completely. Fastest.</description></item>
/// <item><description><b>CommitmentOnly:</b> Clients commit to values before aggregation.
/// Prevents adaptive attacks but doesn't verify correctness.</description></item>
/// <item><description><b>NormBound:</b> Clients prove their gradient norm is within a bound.
/// Prevents gradient scaling attacks.</description></item>
/// <item><description><b>ElementBound:</b> Clients prove each gradient component is bounded.
/// Stronger than norm bound for individual elements.</description></item>
/// <item><description><b>LossThreshold:</b> Clients prove their local loss meets a quality
/// threshold. Detects poorly trained or poisoned models.</description></item>
/// <item><description><b>FullComputation:</b> Clients prove their entire training was done correctly.
/// Strongest guarantee but most expensive (research-stage).</description></item>
/// </list>
/// </remarks>
public enum VerificationLevel
{
    /// <summary>No verification — trust all clients.</summary>
    None,

    /// <summary>Commitment only — clients commit before aggregation to prevent adaptive attacks.</summary>
    CommitmentOnly,

    /// <summary>Norm bound — clients prove gradient L2 norm is within [0, C].</summary>
    NormBound,

    /// <summary>Element bound — clients prove each gradient component is within [-B, B].</summary>
    ElementBound,

    /// <summary>Loss threshold — clients prove local loss is below a threshold.</summary>
    LossThreshold,

    /// <summary>Full computation — clients prove the entire training computation was correct.</summary>
    FullComputation
}
