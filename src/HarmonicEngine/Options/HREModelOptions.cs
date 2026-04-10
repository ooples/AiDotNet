using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.HarmonicEngine.Options;

/// <summary>
/// Configuration options for the Harmonic Resonance Engine model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These options control how the HRE model is built and trained.
/// The HRE replaces traditional neural network layers with spectral communication via orthogonal
/// frequency carriers. Instead of neurons with weights and biases, the HRE uses frequency carriers
/// that interact through intermodulation to compute attention-like scores at O(N log N) complexity.
/// </para>
/// </remarks>
public class HREModelOptions : ModelOptions
{
    /// <summary>
    /// Number of orthogonal frequency carriers used in the OFDM spectral bus.
    /// Each carrier represents one logical feature channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is analogous to the number of neurons in a traditional layer.
    /// Each carrier broadcasts on its own unique frequency, allowing all features to communicate
    /// simultaneously through the spectral bus without point-to-point connections.
    /// More carriers means more features can be represented, but the FFT size must accommodate them.
    /// </para>
    /// </remarks>
    public int CarrierCount { get; set; } = 64;

    /// <summary>
    /// Minimum spacing between adjacent carrier frequencies (in FFT bins).
    /// Larger spacing reduces intermodulation collisions but requires larger FFT sizes.
    /// </summary>
    public int CarrierSpacing { get; set; } = 4;

    /// <summary>
    /// Width of guard bands between carrier groups (in FFT bins).
    /// Guard bands prevent spectral leakage between carrier groups.
    /// </summary>
    public int GuardBandWidth { get; set; } = 2;

    /// <summary>
    /// FFT size for spectral operations. Must be a power of 2.
    /// Determines the frequency resolution and maximum number of carriers.
    /// </summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>
    /// Number of spectral coefficients to retain after sparsity masking (top-K selection).
    /// Lower K provides stronger regularization but may lose signal.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how aggressively the model compresses its internal representation.
    /// Only the K strongest frequency components are kept; the rest are zeroed out.
    /// This acts as a regularizer — preventing overfitting by forcing the model to represent
    /// patterns using only a few dominant frequencies.
    /// Set to 0 to use automatic selection via MDL (Minimum Description Length).
    /// </para>
    /// </remarks>
    public int SparsityK { get; set; } = 16;

    /// <summary>
    /// Whether to use MDL (Minimum Description Length) for automatic K selection.
    /// When true, SparsityK is ignored and K is chosen to minimize description length.
    /// </summary>
    public bool UseMDLAutoK { get; set; }

    /// <summary>
    /// Type of nonlinearity to apply in OFDM layers.
    /// </summary>
    public NonlinearityType Nonlinearity { get; set; } = NonlinearityType.SpectralGating;

    /// <summary>
    /// Type of learning rule for spectral parameter updates.
    /// </summary>
    public LearningRuleType LearningRule { get; set; } = LearningRuleType.Hebbian;

    /// <summary>
    /// Learning rate for the spectral Hebbian update rule.
    /// </summary>
    public double HebbianLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Anti-Hebbian decorrelation strength.
    /// Controls how strongly different frequency components are pushed apart.
    /// </summary>
    public double AntiHebbianAlpha { get; set; } = 0.1;

    /// <summary>
    /// Whether to include the Mellin-Fourier invariance layer as the first stage.
    /// Provides scale and shift invariance for the input signal.
    /// </summary>
    public bool UseMellinFourier { get; set; } = true;

    /// <summary>
    /// Number of OFDM layers in the model.
    /// </summary>
    public int NumOFDMLayers { get; set; } = 2;

    /// <summary>
    /// Number of IMD attention layers in the model.
    /// </summary>
    public int NumAttentionLayers { get; set; } = 1;

    /// <summary>
    /// Output dimension of the model.
    /// </summary>
    public int OutputSize { get; set; } = 1;

    /// <summary>
    /// Input dimension of the model (number of features per time step).
    /// </summary>
    public int InputSize { get; set; } = 1;
}
