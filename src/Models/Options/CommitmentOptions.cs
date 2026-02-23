namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for cryptographic commitment schemes.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A commitment scheme lets you "lock in" a value without revealing it.
/// Later, you can "open" the commitment to prove what the original value was. No one can
/// change the committed value after the fact (binding), and no one can learn the value
/// before you open it (hiding).</para>
/// </remarks>
public class CommitmentOptions
{
    /// <summary>
    /// Gets or sets the hash algorithm for hash-based commitments. Default is "SHA256".
    /// </summary>
    public string HashAlgorithm { get; set; } = "SHA256";

    /// <summary>
    /// Gets or sets the randomness length in bytes for commitment blinding. Default is 32.
    /// </summary>
    public int RandomnessLength { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether to use batch commitments (commit to multiple values at once).
    /// Default is true for efficiency.
    /// </summary>
    public bool UseBatchCommitments { get; set; } = true;

    /// <summary>
    /// Gets or sets the Pedersen generator parameter (prime modulus bit length).
    /// Only used with Pedersen commitments. Default is 256.
    /// </summary>
    public int PedersenGroupBitLength { get; set; } = 256;
}
