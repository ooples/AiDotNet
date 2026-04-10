using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Interfaces;

/// <summary>
/// Interface for spectral-domain learning rules that update frequency-domain filter coefficients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Traditional neural networks learn by backpropagation — computing gradients
/// and adjusting weights iteratively. Spectral learning rules work differently:
/// they update frequency-domain filter coefficients based on the relationship between
/// input and target signals at each frequency.
///
/// Different rules offer different tradeoffs:
/// - Hebbian: biologically plausible, converges in few passes
/// - Wiener: mathematically optimal, single-pass, but only for linear relationships
/// - Phase Alignment: aligns oscillator phases with periodic data patterns
/// </para>
/// </remarks>
public interface ISpectralLearningRule<T>
{
    /// <summary>
    /// Updates the spectral filter based on input and target spectra.
    /// </summary>
    /// <param name="filter">The current spectral filter H(k) — modified in place.</param>
    /// <param name="inputSpectrum">FFT of the input signal X(k).</param>
    /// <param name="targetSpectrum">FFT of the target signal Y(k).</param>
    void Update(Vector<Complex<T>> filter, Vector<Complex<T>> inputSpectrum, Vector<Complex<T>> targetSpectrum);

    /// <summary>
    /// Gets the name of this learning rule for diagnostics.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the current learning rate.
    /// </summary>
    double LearningRate { get; }
}
