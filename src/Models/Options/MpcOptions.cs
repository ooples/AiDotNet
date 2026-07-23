namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for multi-party computation protocols in federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how MPC operates within the FL pipeline.
/// The defaults use additive secret sharing with semi-honest security, which is the fastest
/// configuration suitable for environments where participants are trusted to follow the protocol.</para>
/// </remarks>
public class MpcOptions
{
    /// <summary>
    /// Gets or sets the MPC protocol to use. Default is <see cref="MpcProtocol.AdditiveSecretSharing"/>.
    /// </summary>
    public MpcProtocol Protocol { get; set; } = MpcProtocol.AdditiveSecretSharing;

    /// <summary>
    /// Gets or sets the adversary model. Default is <see cref="MpcSecurityModel.SemiHonest"/>.
    /// </summary>
    public MpcSecurityModel SecurityModel { get; set; } = MpcSecurityModel.SemiHonest;

    /// <summary>
    /// Gets or sets the reconstruction threshold for Shamir-based protocols.
    /// At least this many parties must contribute shares to reconstruct the secret.
    /// Default is 3.
    /// </summary>
    public int Threshold { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of base OTs to perform for OT extension.
    /// Higher values improve security margin. Default is 128.
    /// </summary>
    public int BaseObliviousTransferCount { get; set; } = 128;

    /// <summary>
    /// Gets or sets the batch size for OT extension operations. Default is 1024.
    /// </summary>
    public int ObliviousTransferBatchSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the security parameter in bits (e.g., 128, 256). Default is 128.
    /// </summary>
    public int SecurityParameterBits { get; set; } = 128;

    /// <summary>
    /// Gets or sets the prime field modulus bit length for arithmetic secret sharing.
    /// Default is 64 (using a 64-bit prime for performance).
    /// </summary>
    public int FieldBitLength { get; set; } = 64;

    /// <summary>
    /// Gets or sets the covert security deterrence factor (probability of detecting cheating).
    /// Only used when <see cref="SecurityModel"/> is <see cref="MpcSecurityModel.CovertSecurity"/>.
    /// Default is 0.5 (50% detection probability).
    /// </summary>
    public double CovertDeterrenceFactor { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the gradient clipping norm threshold for secure clipping.
    /// Default is 1.0.
    /// </summary>
    public double ClippingNormThreshold { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to enable free XOR optimization in garbled circuits.
    /// Default is true.
    /// </summary>
    public bool EnableFreeXor { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable half-gates optimization in garbled circuits.
    /// Default is true.
    /// </summary>
    public bool EnableHalfGates { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility. Null for cryptographically secure random.
    /// </summary>
    public int? RandomSeed { get; set; }
}
