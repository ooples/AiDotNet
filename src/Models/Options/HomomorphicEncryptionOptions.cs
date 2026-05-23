namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for homomorphic encryption (HE) in federated learning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Homomorphic encryption lets the server combine encrypted client updates without seeing them in plaintext.
/// This can enable stronger privacy guarantees, at the cost of compute and bandwidth.
/// </remarks>
public class HomomorphicEncryptionOptions : ModelOptions
{
    /// <summary>
    /// Static factory that instantiates a default <c>IHomomorphicEncryptionProvider&lt;T&gt;</c> for a given <c>T</c>
    /// when the federated trainer needs one and the caller did not supply one explicitly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Set this once at startup to provide a fallback provider. The audit-2026-05 phase 2b extraction
    /// moved the SEAL implementation out of the core <c>AiDotNet</c> package into
    /// <c>AiDotNet.Privacy.HE</c>; that package registers itself here via a <c>[ModuleInitializer]</c>
    /// the moment its assembly loads, so applications that simply add the package reference will
    /// continue to work with no startup code changes.
    /// </para>
    /// <para>
    /// Applications that do <i>not</i> reference an HE provider package and <i>do</i> enable HE
    /// (<see cref="Enabled"/> = true) and <i>do not</i> supply an explicit provider on the trainer
    /// constructor will hit a clear <see cref="System.InvalidOperationException"/> at training time
    /// telling them which NuGet to install. This intentional contract change replaces the previous
    /// hard-coded SEAL fallback that pinned <c>Microsoft.Research.SEALNet</c> as a transitive
    /// dependency of every core consumer.
    /// </para>
    /// </remarks>
    public static System.Func<System.Type, object>? DefaultProviderFactory { get; set; }

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
