namespace AiDotNet.HarmonicEngine.Enums;

/// <summary>
/// Specifies the learning rule used to update spectral parameters in Harmonic Resonance Engine layers.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Traditional neural networks learn by backpropagation — computing how much each weight
/// contributed to the error and adjusting it in the opposite direction. The HRE offers alternative learning rules
/// that operate in the frequency domain and can converge in fewer passes (sometimes just one).
/// </para>
/// </remarks>
public enum LearningRuleType
{
    /// <summary>
    /// Spectral Hebbian learning with anti-Hebbian decorrelation.
    /// Updates spectral filter coefficients based on input-output phase coherence.
    /// Converges to the Wiener optimal filter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hebbian learning follows the principle "neurons that fire together wire together."
    /// In the spectral domain, this means: if the input and output signals are in phase (coherent) at a
    /// particular frequency, strengthen that frequency's coupling. If they are out of phase, weaken it.
    /// The anti-Hebbian component prevents all frequencies from converging to the same representation,
    /// forcing the system to capture diverse features.
    /// </para>
    /// </remarks>
    Hebbian,

    /// <summary>
    /// Direct computation of the Wiener optimal linear filter via cross-spectral density.
    /// Computes H_opt(k) = S_xy(k) / S_xx(k) in a single pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Wiener filter is the mathematically optimal linear filter — it minimizes
    /// the mean squared error between the filtered output and the desired signal. Instead of iterating
    /// toward the optimal solution (like gradient descent), this rule computes it directly from the
    /// cross-spectral density (how input and output correlate at each frequency) divided by the
    /// input's power spectrum (how much energy the input has at each frequency).
    /// </para>
    /// </remarks>
    Wiener,

    /// <summary>
    /// Learning by aligning the phase of internal oscillators with the input signal.
    /// Updates carrier phases to maximize correlation with target patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Phase Alignment treats each spectral coefficient as an oscillator with a phase
    /// and amplitude. Learning adjusts the phases to synchronize with recurring patterns in the data.
    /// When the model's internal oscillators are "in sync" with the data's natural rhythms,
    /// prediction accuracy is maximized. This is especially effective for periodic or cyclical data
    /// like financial markets, seasonal patterns, or biological rhythms.
    /// </para>
    /// </remarks>
    PhaseAlignment
}
