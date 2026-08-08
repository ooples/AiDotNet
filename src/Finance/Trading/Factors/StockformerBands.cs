using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.WaveletFunctions;

namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// The dual-frequency band split behind <see cref="Stockformer{T}"/>: a single-level discrete wavelet
/// transform that separates a return series into a LOW-frequency band (trend) and a HIGH-frequency
/// band (short-term fluctuation and abrupt events).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Ma, Xue, Lu and Chen, "Stockformer: A price-volume factor stock selection model based on wavelet
/// transform and multi-task self-attention networks" (arXiv:2401.06139; Expert Systems with
/// Applications 273:126803, 2025).
/// </para>
/// <para>
/// <b>This is a preprocessing stage, not a layer, and that is deliberate.</b> In the reference
/// implementation (github.com/Eric991005/Multitask-Stockformer) the transform lives in the TRAINING
/// SCRIPT — it imports <c>pytorch_wavelets.DWT1DForward</c> and hands the model two already-split
/// inputs, <c>xl</c> and <c>xh</c>. The network never sees the undecomposed series. Implementing the
/// DWT as a layer inside the model would look reasonable and would not match the paper.
/// </para>
/// <para>
/// <b>One level, sym2.</b> From the reference config: <c>wave = sym2</c>, <c>level = 1</c>. Exactly one
/// split into two bands, which is why the encoder downstream has exactly two branches. "Wavelet
/// decomposition" invites building a deep multi-resolution pyramid; the paper does not.
/// </para>
/// <para>
/// sym2 and db2 (Daubechies-2) are the same wavelet — identical four filter taps — so this reuses the
/// library's <see cref="SymletWavelet{T}"/> at order 2 rather than restating the coefficients.
/// </para>
/// <para><b>For Beginners:</b> A stock's return series mixes a slow drift with fast jitter. This
/// separates the two so each can be studied with machinery suited to it — the paper's finding is that
/// treating them identically wastes information.</para>
/// </remarks>
public sealed class StockformerBands<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly SymletWavelet<T> _wavelet;

    /// <summary>Gets the Symlet order in use (2 per the paper).</summary>
    public int WaveletOrder { get; }

    /// <summary>Gets the number of decomposition levels (1 per the paper).</summary>
    public int Levels { get; }

    /// <summary>
    /// Creates a band splitter.
    /// </summary>
    /// <param name="waveletOrder">Symlet order. 2 is the paper's <c>sym2</c>.</param>
    /// <param name="levels">Decomposition levels. 1 is the paper's setting.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="levels"/> is not positive, or when <paramref name="waveletOrder"/>
    /// is not one the underlying Symlet implementation supports.
    /// </exception>
    public StockformerBands(int waveletOrder = 2, int levels = 1)
    {
        if (levels <= 0)
            throw new ArgumentOutOfRangeException(nameof(levels), levels,
                "The number of decomposition levels must be positive.");

        WaveletOrder = waveletOrder;
        Levels = levels;

        // Throws for unsupported orders, which is better than silently substituting a different
        // wavelet than the paper specifies.
        _wavelet = new SymletWavelet<T>(waveletOrder);
    }

    /// <summary>
    /// Splits <paramref name="series"/> into (low, high) bands.
    /// </summary>
    /// <param name="series">A single series over time.</param>
    /// <returns>
    /// The low-frequency approximation and the high-frequency detail. With <see cref="Levels"/> above
    /// one, the low band is split repeatedly and the returned high band is the detail of the LAST,
    /// coarsest level; the finer details produced along the way are discarded because the two-branch
    /// encoder has nowhere to put them.
    /// </returns>
    /// <remarks>
    /// <para>
    /// The coarsest detail is the one returned because it is the only one whose length matches the
    /// low band. Each level halves both, so the level-1 detail has <c>n / 2</c> samples while the low
    /// band after <see cref="Levels"/> splits has <c>n / 2^Levels</c>; pairing those two would hand
    /// the encoder's branches different sequence lengths.
    /// </para>
    /// <para>
    /// At the paper's <c>Levels = 1</c> the coarsest detail IS the finest one, so the distinction only
    /// appears once a caller raises Levels.
    /// </para>
    /// <para>
    /// Each level halves the length, so both bands are shorter than the input. Callers must size the
    /// encoder from <see cref="BandLength"/> rather than from the input window.
    /// </para>
    /// </remarks>
    public (Vector<T> Low, Vector<T> High) Split(Vector<T> series)
    {
        if (series is null) throw new ArgumentNullException(nameof(series));
        if (series.Length < 2)
            throw new ArgumentException(
                $"A series of length {series.Length} cannot be wavelet-decomposed; at least 2 samples " +
                "are required for a single level.", nameof(series));

        var current = series;
        Vector<T> detail = new Vector<T>(0);

        for (int level = 0; level < Levels; level++)
        {
            if (current.Length < 2)
            {
                throw new InvalidOperationException(
                    $"Level {level + 1} of {Levels} cannot be computed: the approximation is down to " +
                    $"{current.Length} sample(s). Reduce Levels or lengthen the input window.");
            }

            (var approximation, var levelDetail) = _wavelet.Decompose(current);
            current = approximation;
            detail = levelDetail;
        }

        return (current, detail);
    }

    /// <summary>
    /// The band length produced from an input window of <paramref name="inputLength"/> samples.
    /// </summary>
    /// <remarks>
    /// Each level halves the length. With the paper's single level and its <c>T1 = 20</c> window, the
    /// bands are 10 long — so the encoder's temporal extent is HALF the input window, which is easy
    /// to get wrong when sizing layers.
    /// </remarks>
    public int BandLength(int inputLength)
    {
        int length = inputLength;
        for (int level = 0; level < Levels; level++) length = length / 2;
        return length;
    }

    /// <summary>
    /// Splits every stock's series in a <c>[stocks, time]</c> matrix, returning one matrix per band.
    /// </summary>
    /// <param name="perStockSeries">Rows are stocks, columns are timesteps.</param>
    /// <returns>Low and high band matrices, each <c>[stocks, BandLength(time)]</c>.</returns>
    public (Matrix<T> Low, Matrix<T> High) SplitAll(Matrix<T> perStockSeries)
    {
        if (perStockSeries is null) throw new ArgumentNullException(nameof(perStockSeries));

        int stocks = perStockSeries.Rows;
        int bandLength = BandLength(perStockSeries.Columns);
        var low = new Matrix<T>(stocks, bandLength);
        var high = new Matrix<T>(stocks, bandLength);

        var row = new Vector<T>(perStockSeries.Columns);
        for (int s = 0; s < stocks; s++)
        {
            for (int t = 0; t < perStockSeries.Columns; t++) row[t] = perStockSeries[s, t];
            (var lowRow, var highRow) = Split(row);

            // Decompose can return a band shorter or longer than the ideal halving depending on how
            // the filter handles the boundary, so copy defensively rather than assuming alignment.
            for (int t = 0; t < bandLength; t++)
            {
                low[s, t] = t < lowRow.Length ? lowRow[t] : Ops.Zero;
                high[s, t] = t < highRow.Length ? highRow[t] : Ops.Zero;
            }
        }

        return (low, high);
    }
}
