namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Defines the abstract interface for a zero-knowledge proof backend.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A ZK proof system provides the low-level cryptographic operations
/// for generating and verifying proofs. Different backends have different tradeoffs:</para>
/// <list type="bullet">
/// <item><description><b>Hash-based:</b> Uses SHA-256. Fast but limited (only commitment proofs).</description></item>
/// <item><description><b>Discrete-log-based:</b> Uses elliptic curves. Supports range proofs and
/// homomorphic operations.</description></item>
/// <item><description><b>Circuit-based (Groth16, PLONK):</b> Supports arbitrary computations by
/// encoding them as arithmetic circuits.</description></item>
/// </list>
/// </remarks>
public interface IZkProofSystem
{
    /// <summary>
    /// Commits to a value using the proof system's commitment scheme.
    /// </summary>
    /// <param name="value">The value to commit to (serialized).</param>
    /// <returns>A tuple of (commitment, randomness used).</returns>
    (byte[] Commitment, byte[] Randomness) Commit(byte[] value);

    /// <summary>
    /// Generates a range proof: proves value is in [0, upperBound].
    /// </summary>
    /// <param name="value">The value (serialized as bytes).</param>
    /// <param name="upperBound">The upper bound (serialized as bytes).</param>
    /// <param name="commitment">The commitment to the value.</param>
    /// <param name="randomness">The randomness used in the commitment.</param>
    /// <returns>The range proof data.</returns>
    byte[] GenerateRangeProof(byte[] value, byte[] upperBound, byte[] commitment, byte[] randomness);

    /// <summary>
    /// Verifies a range proof: checks that the committed value is in [0, upperBound].
    /// </summary>
    /// <param name="proof">The range proof to verify.</param>
    /// <param name="upperBound">The upper bound.</param>
    /// <param name="commitment">The commitment to the value.</param>
    /// <returns>True if the proof is valid.</returns>
    bool VerifyRangeProof(byte[] proof, byte[] upperBound, byte[] commitment);

    /// <summary>
    /// Verifies a commitment opening: checks that the value was indeed committed.
    /// </summary>
    /// <param name="commitment">The commitment.</param>
    /// <param name="value">The claimed value.</param>
    /// <param name="randomness">The randomness used in the commitment.</param>
    /// <returns>True if the opening is valid.</returns>
    bool VerifyOpening(byte[] commitment, byte[] value, byte[] randomness);

    /// <summary>
    /// Gets the name of this proof system.
    /// </summary>
    string Name { get; }
}
