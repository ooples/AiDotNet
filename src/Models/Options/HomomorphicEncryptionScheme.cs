namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies which homomorphic encryption scheme to use.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Different HE schemes support different kinds of math:
/// - CKKS is best for approximate real-number arithmetic.
/// - BFV is best for exact integer arithmetic (often via fixed-point encoding).
/// </remarks>
public enum HomomorphicEncryptionScheme
{
    /// <summary>
    /// CKKS (approximate real numbers).
    /// </summary>
    Ckks = 0,

    /// <summary>
    /// BFV (exact integers).
    /// </summary>
    Bfv = 1
}

