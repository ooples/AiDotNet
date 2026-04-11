namespace AiDotNet.HarmonicEngine.Interfaces;

/// <summary>
/// Interface for frequency carrier allocation strategies.
/// Different allocation strategies trade off carrier count vs. IMD collision order.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When placing signals on frequency carriers, we need to ensure that
/// intermodulation products (created by nonlinearities) don't land on other carriers.
/// Different allocation strategies can guarantee freedom from 2nd-order collisions (basic),
/// 3rd-order collisions (stricter), or provide other tradeoffs.
/// This interface allows swapping allocation strategies without changing the rest of the HRE.
/// </para>
/// </remarks>
public interface ICarrierAllocator
{
    /// <summary>
    /// Allocates carrier frequency bin indices for a given number of carriers and FFT size.
    /// </summary>
    /// <param name="numCarriers">Number of carriers to allocate.</param>
    /// <param name="fftSize">Total FFT size (determines available frequency bins).</param>
    /// <returns>Read-only list of frequency bin indices.</returns>
    IReadOnlyList<int> AllocateCarriers(int numCarriers, int fftSize);

    /// <summary>
    /// Validates that a set of carrier indices has no IMD collisions up to the specified order.
    /// </summary>
    /// <param name="carriers">The carrier frequency bin indices to validate.</param>
    /// <param name="maxOrder">Maximum IMD order to check.</param>
    /// <returns>True if no collisions exist; false otherwise.</returns>
    bool ValidateNoCollisions(IReadOnlyList<int> carriers, int maxOrder = 2);

    /// <summary>
    /// Gets the maximum number of carriers that can be allocated collision-free
    /// for a given FFT size and maximum IMD order.
    /// </summary>
    /// <param name="fftSize">Total FFT size.</param>
    /// <param name="maxOrder">Maximum IMD order to guarantee freedom from.</param>
    /// <returns>Maximum number of collision-free carriers.</returns>
    int MaxCarriers(int fftSize, int maxOrder = 2);
}
