namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the zero-knowledge proof system to use for verifiable federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Zero-knowledge proofs let you prove something is true without
/// revealing why it's true. Different proof systems trade off between speed, proof size,
/// and what they can prove:</para>
/// <list type="bullet">
/// <item><description><b>HashCommitment:</b> Fastest. Uses SHA-256 hashes. Can only prove
/// "I committed to this value" (no arithmetic proofs).</description></item>
/// <item><description><b>Pedersen:</b> Additively homomorphic. The server can verify a sum
/// of committed values without seeing the individual values.</description></item>
/// <item><description><b>Bulletproofs:</b> Range proofs without trusted setup. Can prove
/// "my value is in [0, C]" with logarithmic proof size.</description></item>
/// <item><description><b>Groth16:</b> General-purpose (any computation). Tiny proofs but
/// requires a per-circuit trusted setup.</description></item>
/// <item><description><b>Plonk:</b> General-purpose with universal trusted setup (one setup
/// works for any circuit up to a given size).</description></item>
/// </list>
/// </remarks>
public enum ZkProofSystem
{
    /// <summary>SHA-256 hash commitment — fast, simple, non-interactive. No arithmetic proofs.</summary>
    HashCommitment,

    /// <summary>Pedersen commitment — additively homomorphic. Supports sum verification.</summary>
    Pedersen,

    /// <summary>Bulletproofs — logarithmic-size range proofs without trusted setup.</summary>
    Bulletproofs,

    /// <summary>Groth16 — constant-size proofs for arbitrary computations. Requires per-circuit setup.</summary>
    Groth16,

    /// <summary>PLONK — universal setup, supports arbitrary computations.</summary>
    Plonk
}
