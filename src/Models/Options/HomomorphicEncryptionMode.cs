namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies how homomorphic encryption is applied during federated aggregation.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> HE-only encrypts everything, while hybrid encrypts only selected parameters to reduce cost.
/// </remarks>
public enum HomomorphicEncryptionMode
{
    /// <summary>
    /// Encrypt all parameters for aggregation.
    /// </summary>
    HeOnly = 0,

    /// <summary>
    /// Encrypt only selected parameter ranges; remaining parameters use the normal pipeline.
    /// </summary>
    Hybrid = 1
}

