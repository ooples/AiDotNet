using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Core;

/// <summary>
/// Extracts intermodulation distortion (IMD) products from a signal after nonlinear processing.
/// IMD products encode pairwise feature interactions — the spectral equivalent of attention scores.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When a signal containing multiple frequencies passes through a nonlinear device
/// (like squaring), new frequencies appear at the sums and differences of the original frequencies.
/// These are called intermodulation products.
///
/// For example, if feature A is on carrier f1=10 Hz and feature B is on carrier f2=17 Hz,
/// after squaring, new energy appears at:
/// - f1 + f2 = 27 Hz (sum product) — amplitude proportional to A * B
/// - |f1 - f2| = 7 Hz (difference product) — amplitude also proportional to A * B
///
/// These A * B products are exactly the pairwise interactions we want — they tell us how
/// strongly features A and B are correlated in the input. This is mathematically equivalent
/// to the Q*K^T attention score computation, but computed via FFT at O(N log N) instead of O(N^2).
/// </para>
/// </remarks>
public class IMDExtractor<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly FastFourierTransform<T> _fft;
    private readonly int[] _carriers;
    private readonly int _fftSize;

    // Precomputed IMD product locations for each carrier pair
    private readonly (int sumBin, int diffBin)[,] _imdBins;

    /// <summary>
    /// Initializes a new IMDExtractor with carrier positions.
    /// </summary>
    /// <param name="carriers">Carrier frequency bin indices.</param>
    /// <param name="fftSize">FFT size.</param>
    public IMDExtractor(int[] carriers, int fftSize)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _fft = new FastFourierTransform<T>();
        _carriers = carriers;
        _fftSize = fftSize;

        // Precompute IMD bin locations for all carrier pairs
        int n = carriers.Length;
        _imdBins = new (int, int)[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _imdBins[i, j] = (carriers[i] + carriers[j], Math.Abs(carriers[i] - carriers[j]));
            }
        }
    }

    /// <summary>
    /// Extracts the pairwise interaction matrix from a nonlinearly processed signal.
    /// The (i, j) entry represents the interaction strength between carrier i and carrier j.
    /// </summary>
    /// <param name="nonlinearOutput">Time-domain signal after nonlinear processing.</param>
    /// <returns>N x N interaction matrix where N is the number of carriers.</returns>
    /// <remarks>
    /// <para>
    /// This is the core of the IMD-as-attention mechanism. The interaction matrix is analogous
    /// to the Q*K^T attention score matrix, but computed at O(N log N) via FFT instead of O(N^2).
    /// </para>
    /// </remarks>
    public Matrix<T> ExtractPairwise(Vector<T> nonlinearOutput)
    {
        var spectrum = _fft.Forward(nonlinearOutput);
        int n = _carriers.Length;
        var interactions = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                // Read IMD product amplitude at the sum frequency
                var (sumBin, diffBin) = _imdBins[i, j];

                T interactionStrength;
                if (i == j)
                {
                    // Self-interaction: read at 2*fi (second harmonic)
                    if (sumBin < _fftSize)
                    {
                        interactionStrength = spectrum[sumBin].Magnitude;
                    }
                    else
                    {
                        interactionStrength = _numOps.Zero;
                    }
                }
                else
                {
                    // Cross-interaction: average of sum and difference products
                    T sumMag = sumBin < _fftSize ? spectrum[sumBin].Magnitude : _numOps.Zero;
                    T diffMag = diffBin >= 0 && diffBin < _fftSize ? spectrum[diffBin].Magnitude : _numOps.Zero;
                    interactionStrength = _numOps.Multiply(
                        _numOps.FromDouble(0.5),
                        _numOps.Add(sumMag, diffMag));
                }

                interactions[i, j] = interactionStrength;
                interactions[j, i] = interactionStrength; // Symmetric
            }
        }

        return interactions;
    }

    /// <summary>
    /// Extracts interaction scores as a flattened vector (upper triangle of the interaction matrix).
    /// Useful as input to subsequent layers.
    /// </summary>
    /// <param name="nonlinearOutput">Time-domain signal after nonlinear processing.</param>
    /// <returns>Flattened interaction scores of length N*(N+1)/2.</returns>
    public Vector<T> ExtractFlat(Vector<T> nonlinearOutput)
    {
        var spectrum = _fft.Forward(nonlinearOutput);
        int n = _carriers.Length;
        int flatSize = n * (n + 1) / 2;
        var flat = new Vector<T>(flatSize);
        int idx = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                var (sumBin, diffBin) = _imdBins[i, j];

                if (i == j)
                {
                    flat[idx] = sumBin < _fftSize ? spectrum[sumBin].Magnitude : _numOps.Zero;
                }
                else
                {
                    T sumMag = sumBin < _fftSize ? spectrum[sumBin].Magnitude : _numOps.Zero;
                    T diffMag = diffBin >= 0 && diffBin < _fftSize ? spectrum[diffBin].Magnitude : _numOps.Zero;
                    flat[idx] = _numOps.Multiply(
                        _numOps.FromDouble(0.5),
                        _numOps.Add(sumMag, diffMag));
                }

                idx++;
            }
        }

        return flat;
    }

    /// <summary>
    /// Extracts interaction scores and applies softmax normalization per row,
    /// producing attention weights that sum to 1 for each feature.
    /// </summary>
    /// <param name="nonlinearOutput">Time-domain signal after nonlinear processing.</param>
    /// <returns>N x N attention weight matrix (rows sum to 1).</returns>
    public Matrix<T> ExtractAttentionWeights(Vector<T> nonlinearOutput)
    {
        var interactions = ExtractPairwise(nonlinearOutput);
        int n = _carriers.Length;
        var weights = new Matrix<T>(n, n);

        // Softmax per row
        for (int i = 0; i < n; i++)
        {
            // Find max for numerical stability
            T maxVal = interactions[i, 0];
            for (int j = 1; j < n; j++)
            {
                var val = interactions[i, j];
                if (_numOps.ToDouble(val) > _numOps.ToDouble(maxVal))
                    maxVal = val;
            }

            // Compute exp(x - max) and sum
            T sumExp = _numOps.Zero;
            for (int j = 0; j < n; j++)
            {
                var expVal = _numOps.FromDouble(
                    Math.Exp(_numOps.ToDouble(_numOps.Subtract(interactions[i, j], maxVal))));
                weights[i, j] = expVal;
                sumExp = _numOps.Add(sumExp, expVal);
            }

            // Normalize
            for (int j = 0; j < n; j++)
            {
                weights[i, j] = _numOps.Divide(weights[i, j], sumExp);
            }
        }

        return weights;
    }
}
