using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified;

/// <summary>
/// Stratified split for regression targets that preserves the target distribution.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Stratification is usually for classification (preserving class counts).
/// But what about regression where targets are continuous?
/// </para>
/// <para>
/// <b>Solution:</b> We bin the continuous target into groups, then stratify by those bins.
/// This ensures both train and test have similar target distributions.
/// </para>
/// <para>
/// <b>Example:</b>
/// If house prices range from $100k-$1M, we might create 10 bins:
/// $100k-$190k, $190k-$280k, etc.
/// Then ensure each split has proportional samples from each bin.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Regression problems with skewed target distributions
/// - When you want representative samples across the target range
/// - Prevents scenarios where all expensive houses end up in test
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DistributionPreservingSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly int _nBins;

    /// <summary>
    /// Creates a new distribution-preserving splitter for regression.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="nBins">Number of bins for stratification. Default is 10.</param>
    /// <param name="shuffle">Whether to shuffle within bins. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DistributionPreservingSplitter(
        double testSize = 0.2,
        int nBins = 10,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (nBins < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(nBins), "Number of bins must be at least 2.");
        }

        _testSize = testSize;
        _nBins = nBins;
    }

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Distribution-Preserving split ({_nBins} bins, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y), "Distribution-preserving split requires target values (y).");
        }

        // Convert targets to double for binning
        var values = new double[y.Length];
        for (int i = 0; i < y.Length; i++)
        {
            values[i] = Convert.ToDouble(y[i]);
        }

        // Calculate bin edges
        double minVal = values.Min();
        double maxVal = values.Max();
        double binWidth = (maxVal - minVal) / _nBins;

        // Handle edge case where all values are the same
        if (binWidth == 0)
        {
            // Fall back to regular random split
            var indices = GetShuffledIndices(X.Rows);
            int testSize = Math.Max(1, (int)(X.Rows * _testSize));
            int trainSize = X.Rows - testSize;
            return BuildResult(X, y, indices.Take(trainSize).ToArray(), indices.Skip(trainSize).ToArray());
        }

        // Assign each sample to a bin
        var bins = new Dictionary<int, List<int>>();
        for (int i = 0; i < y.Length; i++)
        {
            int bin = Math.Min(_nBins - 1, (int)((values[i] - minVal) / binWidth));
            if (!bins.TryGetValue(bin, out var list))
            {
                list = new List<int>();
                bins[bin] = list;
            }
            list.Add(i);
        }

        // Stratified split by bins
        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        foreach (var bin in bins.Values)
        {
            var binIndices = bin.ToArray();
            if (_shuffle)
            {
                ShuffleIndices(binIndices);
            }

            int binTestSize = Math.Max(1, (int)(binIndices.Length * _testSize));
            int binTrainSize = binIndices.Length - binTestSize;

            for (int i = 0; i < binTrainSize; i++)
            {
                trainIndices.Add(binIndices[i]);
            }

            for (int i = binTrainSize; i < binIndices.Length; i++)
            {
                testIndices.Add(binIndices[i]);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
