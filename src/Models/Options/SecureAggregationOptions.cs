using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Privacy;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for secure aggregation in federated learning (Bonawitz et al. 2017).
/// </summary>
/// <remarks>
/// <para>
/// The protocol is "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (Bonawitz, Ivanov, Kreuter,
/// Marcedone, McMahan, Patel, Ramage, Segal and Seth; CCS 2017), honest-but-curious version. Client updates are
/// fixed-point encoded into the ring <c>Z_R</c>, <c>R = 2^BitWidth</c>, masked with pairwise and self masks that cancel
/// exactly in modular arithmetic, and the server recovers the exact sum of the survivors' encodings as long as at least
/// <see cref="ReconstructionThreshold"/> clients complete every round.
/// </para>
/// <para><b>For Beginners:</b> Secure aggregation hides each client's update from the server so the server
/// can only see the final combined result. These options pick how many clients must stay connected for a round to
/// succeed and how precisely model values are represented while they are masked. The defaults work for typical
/// models; change them only if your parameters exceed <see cref="ClipBound"/> in magnitude or you need more
/// precision.</para>
/// </remarks>
public class SecureAggregationOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets whether secure aggregation is enabled.
    /// </summary>
    /// <remarks>
    /// This is an alternative to <see cref="FederatedLearningOptions.UseSecureAggregation"/>. If either is enabled,
    /// secure aggregation is used.
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets which secure aggregation mode is used.
    /// </summary>
    /// <remarks>
    /// <see cref="SecureAggregationMode.FullParticipation"/> uses threshold <c>t = n</c>;
    /// <see cref="SecureAggregationMode.ThresholdDropoutResilient"/> uses <see cref="ReconstructionThreshold"/>.
    /// </remarks>
    public SecureAggregationMode Mode { get; set; } = SecureAggregationMode.FullParticipation;

    /// <summary>
    /// Gets or sets the Shamir threshold <c>t</c> used by <see cref="SecureAggregationMode.ThresholdDropoutResilient"/>.
    /// </summary>
    /// <remarks>
    /// At least <c>t</c> clients must complete every round, so up to <c>n - t</c> may drop. If this is 0 or less, the
    /// default <c>floor(2n/3) + 1</c> is used: the bound <c>t &gt; 2n/3</c> of the paper's security analysis. A
    /// configured value must be between 2 and the number of selected clients.
    /// </remarks>
    public int ReconstructionThreshold { get; set; } = 0;

    /// <summary>
    /// Gets or sets the ring bit width <c>b</c>; masked values live in <c>Z_R</c> with <c>R = 2^b</c>.
    /// </summary>
    /// <remarks>
    /// Between 2 and 64. The sum of <c>n</c> clipped inputs must stay below <c>2^(b-1)</c> in magnitude:
    /// <c>n * ClipBound * 2^FractionalBits &lt; 2^(b-1)</c>, which is checked when a round starts.
    /// </remarks>
    public int BitWidth { get; set; } = FixedPointRingEncoding.DefaultBitWidth;

    /// <summary>
    /// Gets or sets the number of fractional bits <c>f</c> of the fixed-point encoding (scale <c>2^f</c>).
    /// </summary>
    /// <remarks>
    /// The default 24 resolves steps of about 6e-8 and represents every half-precision value exactly.
    /// </remarks>
    public int FractionalBits { get; set; } = FixedPointRingEncoding.DefaultFractionalBits;

    /// <summary>
    /// Gets or sets the clipping bound <c>C</c>: each (weighted) parameter is clipped to <c>[-C, C]</c> before encoding.
    /// </summary>
    public double ClipBound { get; set; } = FixedPointRingEncoding.DefaultClipBound;

    /// <summary>
    /// Gets or sets the pseudorandom generator that expands mask seeds; <c>null</c> selects
    /// <see cref="AesCtrSecureAggregationPrg"/> (AES-256 in counter mode).
    /// </summary>
    public ISecureAggregationPrg? Prg { get; set; } = null;
}
