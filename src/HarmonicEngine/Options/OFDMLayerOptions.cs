using AiDotNet.HarmonicEngine.Enums;

namespace AiDotNet.HarmonicEngine.Options;

/// <summary>
/// Configuration options for an individual OFDM (Orthogonal Frequency Division Multiplexing) layer.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Each OFDM layer encodes input features onto orthogonal frequency carriers,
/// applies a nonlinear activation in the spectral domain, and decodes the result.
/// This replaces the traditional dense layer's matrix multiplication (W * x + b) with a spectral
/// broadcast-and-interfere operation that costs O(N log N) instead of O(N^2).
/// </para>
/// </remarks>
public class OFDMLayerOptions
{
    /// <summary>
    /// Number of orthogonal frequency carriers in this layer.
    /// </summary>
    public int NumCarriers { get; set; } = 64;

    /// <summary>
    /// Spacing between carrier frequency bins to prevent spectral overlap.
    /// </summary>
    public int CarrierSpacing { get; set; } = 4;

    /// <summary>
    /// Ratio of guard band width to carrier spacing.
    /// A value of 0.25 means 25% of the spacing is reserved as guard band.
    /// </summary>
    public double GuardBandRatio { get; set; } = 0.25;

    /// <summary>
    /// Length of the cyclic prefix (in samples) to prevent inter-symbol interference.
    /// Set to 0 to disable cyclic prefix.
    /// </summary>
    public int CyclicPrefixLength { get; set; }

    /// <summary>
    /// FFT size for this layer. Must be a power of 2.
    /// </summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>
    /// Type of nonlinearity to apply after carrier encoding.
    /// </summary>
    public NonlinearityType Nonlinearity { get; set; } = NonlinearityType.SpectralGating;
}
