namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the cryptographic protocol used for Private Set Intersection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different PSI protocols trade off speed, security, and functionality.
/// Think of these as different methods of comparing guest lists privately:</para>
/// <list type="bullet">
/// <item><description><b>DiffieHellman:</b> Simple, well-understood, good for small-to-medium sets.</description></item>
/// <item><description><b>ObliviousTransfer:</b> Fastest for large sets, uses cuckoo hashing for efficiency.</description></item>
/// <item><description><b>CircuitBased:</b> Most flexible, supports computing functions on the intersection (e.g., sum, count).</description></item>
/// <item><description><b>BloomFilter:</b> Probabilistic, fastest but with a small false-positive rate.</description></item>
/// </list>
/// </remarks>
public enum PsiProtocol
{
    /// <summary>
    /// Diffie-Hellman based PSI using commutative encryption.
    /// Both parties hash and re-encrypt each other's sets, then compare.
    /// O(n) communication, simple implementation, well-understood security.
    /// </summary>
    DiffieHellman = 0,

    /// <summary>
    /// Oblivious Transfer based PSI using cuckoo hashing.
    /// Faster for large sets with sublinear communication in some variants.
    /// </summary>
    ObliviousTransfer = 1,

    /// <summary>
    /// Circuit-based PSI using garbled circuits or secret sharing.
    /// Supports arbitrary functions on the intersection (not just membership).
    /// </summary>
    CircuitBased = 2,

    /// <summary>
    /// Bloom filter based probabilistic PSI.
    /// Very fast with small communication overhead but has a configurable false-positive rate.
    /// </summary>
    BloomFilter = 3
}
