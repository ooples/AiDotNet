using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Interfaces;

/// <summary>
/// Extends <see cref="ILayer{T}"/> with spectral-domain-specific operations for HRE layers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Traditional neural network layers operate on raw values (vectors, matrices).
/// HRE layers operate in the frequency domain — they encode, transform, and decode spectral representations.
/// This interface adds methods that let you inspect and manipulate the spectral state of a layer:
/// the carrier frequencies it uses, the current spectrum, and the spectral filter coefficients.
/// </para>
/// </remarks>
public interface ISpectralLayer<T> : ILayer<T>
{
    /// <summary>
    /// Gets the carrier frequency bin indices used by this layer.
    /// </summary>
    int[] GetCarrierBins();

    /// <summary>
    /// Gets the FFT size used by this layer for spectral operations.
    /// </summary>
    int FftSize { get; }

    /// <summary>
    /// Gets the number of frequency carriers in this layer.
    /// </summary>
    int NumCarriers { get; }

    /// <summary>
    /// Gets the current spectral filter coefficients (if the layer has a learnable filter).
    /// Returns null for layers without learnable spectral parameters.
    /// </summary>
    Vector<Complex<T>>? GetSpectralFilter();

    /// <summary>
    /// Gets the power spectrum of the most recent input processed by this layer.
    /// Useful for diagnostics and visualization.
    /// </summary>
    Vector<T>? GetLastInputPowerSpectrum();
}
