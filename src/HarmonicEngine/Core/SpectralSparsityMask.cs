using AiDotNet.HarmonicEngine.Core;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Core;

/// <summary>
/// Applies top-K frequency selection as a spectral sparsity mask, retaining only the K strongest
/// frequency components and zeroing the rest. This provides principled regularization backed by
/// compressed sensing theory.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Most real-world signals can be represented by just a few dominant frequencies
/// — the rest is noise. Spectral sparsity exploits this by keeping only the top-K strongest
/// frequency components and throwing away the rest.
///
/// This serves two purposes:
/// 1. Regularization: prevents overfitting by limiting model capacity (like dropout for frequencies)
/// 2. Compression: the model only stores K coefficients instead of N, dramatically reducing model size
///
/// The Minimum Description Length (MDL) principle can automatically choose the optimal K:
/// it balances signal fidelity (more K = better fit) against model complexity (more K = more to store).
/// </para>
/// </remarks>
public class SpectralSparsityMask<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the SpectralSparsityMask class.
    /// </summary>
    public SpectralSparsityMask()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Applies top-K sparsity to a complex spectrum, retaining only the K components with largest magnitude.
    /// </summary>
    /// <param name="spectrum">The input complex spectrum.</param>
    /// <param name="k">Number of components to retain.</param>
    /// <returns>The sparse spectrum with only K non-zero components.</returns>
    public Vector<Complex<T>> Apply(Vector<Complex<T>> spectrum, int k)
    {
        int n = spectrum.Length;
        if (n == 0) return spectrum;
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "K must be positive.");
        k = Math.Min(k, n);

        // Engine-accelerated TopK by magnitude
        var tensorSpec = SpectralEngineHelper.ToComplexTensor(spectrum);
        var sparseResult = SpectralEngineHelper.TopK(tensorSpec, k);
        return SpectralEngineHelper.ToComplexVector(sparseResult);
    }

    /// <summary>
    /// Returns the indices of the top-K components by magnitude.
    /// </summary>
    /// <param name="spectrum">The input complex spectrum.</param>
    /// <param name="k">Number of top components to select.</param>
    /// <returns>Array of frequency bin indices for the K strongest components.</returns>
    public int[] GetTopKIndices(Vector<Complex<T>> spectrum, int k)
    {
        int n = spectrum.Length;
        if (n == 0) return [];
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "K must be positive.");
        k = Math.Min(k, n);

        var magnitudes = new (double magnitude, int index)[n];
        for (int i = 0; i < n; i++)
        {
            magnitudes[i] = (_numOps.ToDouble(spectrum[i].Magnitude), i);
        }

        Array.Sort(magnitudes, (a, b) => b.magnitude.CompareTo(a.magnitude));

        var indices = new int[k];
        for (int i = 0; i < k; i++)
        {
            indices[i] = magnitudes[i].index;
        }

        return indices;
    }

    /// <summary>
    /// Automatically selects the optimal K using the Minimum Description Length (MDL) principle.
    /// MDL balances signal fidelity against model complexity.
    /// </summary>
    /// <param name="spectrum">The input complex spectrum.</param>
    /// <returns>The optimal number of components K that minimizes description length.</returns>
    /// <remarks>
    /// <para>
    /// MDL cost = reconstruction_error(K) + K * log(N) * bits_per_coefficient
    ///
    /// As K increases, reconstruction error decreases but model cost increases.
    /// The optimal K is where the total cost is minimized.
    /// </para>
    /// </remarks>
    public int SelectK(Vector<Complex<T>> spectrum)
    {
        int n = spectrum.Length;
        if (n == 0) return 0;

        // Compute sorted magnitudes (descending)
        var magnitudes = new double[n];
        double totalEnergy = 0;
        for (int i = 0; i < n; i++)
        {
            magnitudes[i] = _numOps.ToDouble(spectrum[i].Magnitude);
            totalEnergy += magnitudes[i] * magnitudes[i];
        }

        Array.Sort(magnitudes);
        Array.Reverse(magnitudes); // Descending

        if (totalEnergy < 1e-15)
            return 1; // Signal is essentially zero

        double logN = Math.Log(n);
        double bestCost = double.MaxValue;
        int bestK = 1;
        double capturedEnergy = 0;

        for (int k = 1; k <= n; k++)
        {
            capturedEnergy += magnitudes[k - 1] * magnitudes[k - 1];
            double residualEnergy = totalEnergy - capturedEnergy;

            // Reconstruction error (normalized)
            double reconstructionCost = residualEnergy / totalEnergy;

            // Model complexity cost (MDL penalty)
            double complexityCost = (double)k * logN / n;

            double totalCost = reconstructionCost + complexityCost;

            if (totalCost < bestCost)
            {
                bestCost = totalCost;
                bestK = k;
            }
        }

        return bestK;
    }

    /// <summary>
    /// Computes the fraction of total spectral energy captured by the top-K components.
    /// </summary>
    /// <param name="spectrum">The input complex spectrum.</param>
    /// <param name="k">Number of components.</param>
    /// <returns>Energy ratio in [0, 1]. A value of 0.95 means 95% of energy is captured.</returns>
    public double EnergyRatio(Vector<Complex<T>> spectrum, int k)
    {
        int n = spectrum.Length;
        k = Math.Min(k, n);

        var magnitudes = new double[n];
        double totalEnergy = 0;
        for (int i = 0; i < n; i++)
        {
            double mag = _numOps.ToDouble(spectrum[i].Magnitude);
            magnitudes[i] = mag * mag;
            totalEnergy += magnitudes[i];
        }

        if (totalEnergy < 1e-15)
            return 1.0;

        Array.Sort(magnitudes);
        Array.Reverse(magnitudes);

        double capturedEnergy = 0;
        for (int i = 0; i < k; i++)
        {
            capturedEnergy += magnitudes[i];
        }

        return capturedEnergy / totalEnergy;
    }
}
