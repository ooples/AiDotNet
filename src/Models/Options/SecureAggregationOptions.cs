using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Privacy;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for secure aggregation in federated learning (Bonawitz et al. 2017).
/// </summary>
/// <remarks>
/// <para><b>Reference:</b> K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage,
/// A. Segal and K. Seth, "Practical Secure Aggregation for Privacy-Preserving Machine Learning", ACM CCS 2017
/// (honest-but-curious version).</para>
/// <para>
/// Client updates are fixed-point encoded into the ring <c>Z_R</c>, <c>R = 2^BitWidth</c>, masked with pairwise and
/// self masks that cancel exactly in modular arithmetic, and the server recovers the exact sum of the survivors'
/// encodings as long as at least <see cref="ReconstructionThreshold"/> clients complete every round.
/// </para>
/// <para><b>Model size:</b> the masked vector has one ring element per model parameter, so each client sends
/// <c>8 * parameterCount</c> bytes per round at the default 64-bit width, plus <c>O(n)</c> key shares; mask
/// expansion costs one PRG stream of that length per pair of clients. The parameter count does not affect the
/// overflow bound, but the client count does: <c>n * round(ClipBound * 2^FractionalBits)</c> must stay below
/// <c>2^(BitWidth-1)</c>, so the defaults (64 bits, 24 fractional bits, clip 65536) fit fewer than 2^23 clients.
/// For more clients, lower <see cref="ClipBound"/> or <see cref="FractionalBits"/>.</para>
/// <para><b>For Beginners:</b> Secure aggregation hides each client's update from the server so the server
/// can only see the final combined result. These options pick how many clients must stay connected for a round to
/// succeed and how precisely model values are represented while they are masked. The defaults work for typical
/// models; change them only if your parameters exceed <see cref="ClipBound"/> in magnitude or you need more
/// precision.</para>
/// </remarks>
public class SecureAggregationOptions : ModelOptions
{
    /// <summary>Initializes a new instance with the defaults: disabled, full participation, the default encoding.</summary>
    public SecureAggregationOptions()
    {
    }

    /// <summary>Initializes a new instance by copying every setting from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public SecureAggregationOptions(SecureAggregationOptions other)
        : base(other)
    {
        Enabled = other.Enabled;
        Mode = other.Mode;
        ReconstructionThreshold = other.ReconstructionThreshold;
        BitWidth = other.BitWidth;
        FractionalBits = other.FractionalBits;
        ClipBound = other.ClipBound;
        Prg = other.Prg;
    }

    /// <summary>
    /// Gets or sets whether secure aggregation is enabled.
    /// </summary>
    /// <value>Defaults to <c>false</c>.</value>
    /// <remarks>
    /// <para>This is an alternative to <see cref="FederatedLearningOptions.UseSecureAggregation"/>. If either is
    /// enabled, secure aggregation is used.</para>
    /// <para><b>For Beginners:</b> Turn this on to keep individual client updates hidden from the server.</para>
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets which secure aggregation mode is used.
    /// </summary>
    /// <value>Defaults to <see cref="SecureAggregationMode.FullParticipation"/>.</value>
    /// <remarks>
    /// <para><see cref="SecureAggregationMode.FullParticipation"/> uses threshold <c>t = n</c>;
    /// <see cref="SecureAggregationMode.ThresholdDropoutResilient"/> uses <see cref="ReconstructionThreshold"/>.</para>
    /// <para><b>For Beginners:</b> Full participation fails the round if any client disconnects; the dropout-resilient
    /// mode still finishes as long as enough clients stay.</para>
    /// </remarks>
    public SecureAggregationMode Mode { get; set; } = SecureAggregationMode.FullParticipation;

    /// <summary>
    /// Gets or sets the Shamir threshold <c>t</c> used by <see cref="SecureAggregationMode.ThresholdDropoutResilient"/>.
    /// </summary>
    /// <value>Defaults to 0, which selects <c>floor(2n/3) + 1</c>.</value>
    /// <remarks>
    /// <para>At least <c>t</c> clients must complete every round, so up to <c>n - t</c> may drop. If this is 0 or
    /// less, the default <c>floor(2n/3) + 1</c> is used: the bound <c>t &gt; 2n/3</c> of the paper's security
    /// analysis. A configured value must be between 2 and the number of selected clients.</para>
    /// <para><b>For Beginners:</b> This is how many clients must stay connected. A lower value tolerates more
    /// dropouts but needs fewer colluding parties to reveal an individual update.</para>
    /// </remarks>
    public int ReconstructionThreshold { get; set; } = 0;

    /// <summary>
    /// Gets or sets the ring bit width <c>b</c>; masked values live in <c>Z_R</c> with <c>R = 2^b</c>.
    /// </summary>
    /// <value>Defaults to <see cref="FixedPointRingEncoding.DefaultBitWidth"/> (64).</value>
    /// <remarks>
    /// <para>Between 2 and 64. The sum of <c>n</c> clipped, rounded inputs must stay below <c>2^(b-1)</c> in
    /// magnitude: <c>n * round(ClipBound * 2^FractionalBits) &lt; 2^(b-1)</c>, which is checked when a round
    /// starts.</para>
    /// <para><b>For Beginners:</b> This is the size of the whole numbers the masked values are stored in. Bigger
    /// numbers leave room for more clients and more precision.</para>
    /// </remarks>
    public int BitWidth { get; set; } = FixedPointRingEncoding.DefaultBitWidth;

    /// <summary>
    /// Gets or sets the number of fractional bits <c>f</c> of the fixed-point encoding (scale <c>2^f</c>).
    /// </summary>
    /// <value>Defaults to <see cref="FixedPointRingEncoding.DefaultFractionalBits"/> (24).</value>
    /// <remarks>
    /// <para>The default 24 resolves steps of about 6e-8 and represents every half-precision value exactly.</para>
    /// <para><b>For Beginners:</b> This sets how many binary digits after the point are kept. Each value is rounded
    /// to a multiple of <c>2^-f</c> before masking.</para>
    /// </remarks>
    public int FractionalBits { get; set; } = FixedPointRingEncoding.DefaultFractionalBits;

    /// <summary>
    /// Gets or sets the clipping bound <c>C</c>: each (weighted) parameter is clipped to <c>[-C, C]</c> before encoding.
    /// </summary>
    /// <value>Defaults to <see cref="FixedPointRingEncoding.DefaultClipBound"/> (65536).</value>
    /// <remarks>
    /// <para>The bound applies after the client weight. <c>round(C * 2^FractionalBits)</c> must be below
    /// <c>2^(BitWidth-1)</c>.</para>
    /// <para><b>For Beginners:</b> Values outside this range are replaced by the nearest boundary before encoding,
    /// which changes the aggregate when an input exceeds it. Keep it above the largest weighted parameter.</para>
    /// </remarks>
    public double ClipBound { get; set; } = FixedPointRingEncoding.DefaultClipBound;

    /// <summary>
    /// Gets or sets the pseudorandom generator that expands mask seeds.
    /// </summary>
    /// <value>Defaults to <c>null</c>, which selects <see cref="AesCtrSecureAggregationPrg"/> (AES-256 in counter
    /// mode).</value>
    /// <remarks>
    /// <para>Every client and the server must use the same generator, or the masks do not cancel.</para>
    /// <para><b>For Beginners:</b> This turns a short secret into the long random mask. Leave it unset unless you
    /// need a specific generator.</para>
    /// </remarks>
    public ISecureAggregationPrg? Prg { get; set; } = null;
}
