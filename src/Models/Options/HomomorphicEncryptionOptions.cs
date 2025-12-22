namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for homomorphic encryption (HE) in federated learning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Homomorphic encryption lets the server combine encrypted client updates without seeing them in plaintext.
/// This can enable stronger privacy guarantees, at the cost of compute and bandwidth.
/// </remarks>
public class HomomorphicEncryptionOptions
{
    /// <summary>
    /// Gets or sets whether homomorphic encryption is enabled.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the HE scheme.
    /// </summary>
    public HomomorphicEncryptionScheme Scheme { get; set; } = HomomorphicEncryptionScheme.Ckks;

    /// <summary>
    /// Gets or sets the HE mode.
    /// </summary>
    public HomomorphicEncryptionMode Mode { get; set; } = HomomorphicEncryptionMode.HeOnly;

    /// <summary>
    /// Gets or sets which parameter ranges are encrypted when <see cref="Mode"/> is <see cref="HomomorphicEncryptionMode.Hybrid"/>.
    /// </summary>
    public List<ParameterIndexRange> EncryptedRanges { get; set; } = new List<ParameterIndexRange>();

    /// <summary>
    /// Gets or sets the polynomial modulus degree.
    /// </summary>
    /// <remarks>
    /// Typical values: 4096, 8192, 16384 (larger is more secure but slower).
    /// </remarks>
    public int PolyModulusDegree { get; set; } = 8192;

    /// <summary>
    /// Gets or sets CKKS coefficient modulus bit sizes.
    /// </summary>
    /// <remarks>
    /// This controls precision and security for CKKS. If empty, a safe default is used.
    /// </remarks>
    public List<int> CkksCoeffModulusBits { get; set; } = new List<int>();

    /// <summary>
    /// Gets or sets CKKS scale used for encoding.
    /// </summary>
    public double CkksScale { get; set; } = Math.Pow(2.0, 40);

    /// <summary>
    /// Gets or sets BFV plain modulus bit size (batching-friendly prime).
    /// </summary>
    public int BfvPlainModulusBitSize { get; set; } = 40;

    /// <summary>
    /// Gets or sets the fixed-point scaling factor for BFV encoding.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> BFV works with integers, so real numbers are multiplied by this scale and rounded to an integer.
    /// </remarks>
    public double BfvFixedPointScale { get; set; } = 1_000.0;
}
