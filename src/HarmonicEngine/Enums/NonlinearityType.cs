namespace AiDotNet.HarmonicEngine.Enums;

/// <summary>
/// Specifies the type of nonlinear activation used in Harmonic Resonance Engine layers.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In the HRE architecture, nonlinearities operate in the spectral (frequency) domain
/// rather than on individual neuron outputs. These functions create the intermodulation products that enable
/// cross-feature interactions — the spectral equivalent of attention in traditional neural networks.
/// </para>
/// </remarks>
public enum NonlinearityType
{
    /// <summary>
    /// Modified Rectified Linear Unit for complex-valued signals.
    /// Thresholds based on magnitude while preserving phase: f(z) = z * ReLU(|z| + b) / |z|.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ModReLU is the complex-number version of the popular ReLU activation.
    /// Instead of zeroing negative real numbers, it zeroes complex numbers whose magnitude is too small.
    /// The phase (angle) of the complex number is always preserved, so directional information is never lost.
    /// The learnable bias parameter b controls the threshold.
    /// </para>
    /// <para>Reference: Arjovsky et al., "Unitary Evolution Recurrent Neural Networks" (2016).</para>
    /// </remarks>
    ModReLU,

    /// <summary>
    /// Input-dependent spectral mask that learns which frequency-phase combinations to pass.
    /// Each frequency bin has a learnable gate: output[k] = input[k] * sigmoid(W * |input[k]| + b).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spectral Gating works like a smart filter that learns which frequencies matter.
    /// For each frequency bin, it computes a gate value between 0 (block) and 1 (pass) based on the
    /// input signal's magnitude at that frequency. This is adaptive — the gate changes based on the input,
    /// making it a nonlinear operation even though each gate is applied multiplicatively.
    /// Analogous to Gated Linear Units (GLU) but operating in the frequency domain.
    /// </para>
    /// </remarks>
    SpectralGating,

    /// <summary>
    /// Nonlinearity based on instantaneous frequency (derivative of phase) of the analytic signal.
    /// Captures rate of change of oscillatory patterns via the Hilbert transform.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instantaneous Frequency measures how fast the phase of a signal is changing
    /// at each moment in time. This is computed using the Hilbert transform to create the "analytic signal"
    /// (a complex version of the real input), then taking the derivative of its phase.
    /// This nonlinearity is especially powerful for time-series data because it directly encodes
    /// whether oscillations are speeding up or slowing down — information that traditional activations miss entirely.
    /// </para>
    /// </remarks>
    InstantaneousFreq
}
