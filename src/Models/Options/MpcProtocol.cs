namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the multi-party computation protocol to use.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MPC lets multiple parties compute a function together without any
/// party revealing its private input. Different protocols trade off speed for generality:</para>
/// <list type="bullet">
/// <item><description><b>AdditiveSecretSharing:</b> Fastest for linear operations (sums, weighted averages).</description></item>
/// <item><description><b>ShamirSecretSharing:</b> Threshold-based — tolerates party dropouts.</description></item>
/// <item><description><b>GarbledCircuits:</b> Supports arbitrary computations but is slower.</description></item>
/// <item><description><b>Hybrid:</b> Uses additive SS for linear ops and garbled circuits for non-linear ops.</description></item>
/// </list>
/// </remarks>
public enum MpcProtocol
{
    /// <summary>
    /// Additive secret sharing — fast for linear operations (add, scalar multiply).
    /// </summary>
    AdditiveSecretSharing,

    /// <summary>
    /// Shamir secret sharing — threshold-based, tolerates dropouts.
    /// </summary>
    ShamirSecretSharing,

    /// <summary>
    /// Yao's garbled circuits — supports arbitrary boolean/arithmetic computations.
    /// </summary>
    GarbledCircuits,

    /// <summary>
    /// Hybrid: additive SS for linear ops + garbled circuits for non-linear ops (compare, clip).
    /// </summary>
    Hybrid
}
