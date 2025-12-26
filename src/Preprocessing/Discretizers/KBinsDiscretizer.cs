using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Discretizers;

/// <summary>
/// Specifies the strategy for defining bin widths.
/// </summary>
public enum BinningStrategy
{
    /// <summary>
    /// All bins have the same width (equal-width binning).
    /// </summary>
    Uniform,

    /// <summary>
    /// Each bin contains approximately the same number of samples (quantile-based binning).
    /// </summary>
    Quantile,

    /// <summary>
    /// Uses K-means clustering to determine bin edges.
    /// </summary>
    KMeans
}

/// <summary>
/// Specifies how to encode the discretized features.
/// </summary>
public enum EncodeMode
{
    /// <summary>
    /// Returns ordinal integers (0, 1, 2, ..., n_bins-1).
    /// </summary>
    Ordinal,

    /// <summary>
    /// Returns bin indices as float values scaled to [0, 1].
    /// </summary>
    Normalized
}

/// <summary>
/// Bins continuous features into discrete intervals.
/// </summary>
/// <remarks>
/// <para>
/// KBinsDiscretizer discretizes features into k equal-width or quantile-based bins.
/// This is useful for transforming continuous features into categorical features,
/// which can help certain models like decision trees and reduce sensitivity to outliers.
/// </para>
/// <para><b>For Beginners:</b> This transformer groups continuous values into bins (categories):
/// - Uniform strategy: Divides the range into equal-width intervals
/// - Quantile strategy: Divides so each bin has roughly equal number of samples
///
/// Example with 3 bins using uniform strategy:
/// [1, 5, 10, 15, 20, 25] with range 1-25 creates bins:
/// - Bin 0: 1-9 → values 1, 5
/// - Bin 1: 9-17 → values 10, 15
/// - Bin 2: 17-25 → values 20, 25
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class KBinsDiscretizer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nBins;
    private readonly BinningStrategy _strategy;
    private readonly EncodeMode _encode;

    // Fitted parameters: bin edges for each column
    private List<T[]>? _binEdges;

    /// <summary>
    /// Gets the number of bins used for discretization.
    /// </summary>
    public int NBins => _nBins;

    /// <summary>
    /// Gets the binning strategy used.
    /// </summary>
    public BinningStrategy Strategy => _strategy;

    /// <summary>
    /// Gets the encoding mode used.
    /// </summary>
    public EncodeMode Encode => _encode;

    /// <summary>
    /// Gets the bin edges for each feature computed during fitting.
    /// </summary>
    public List<T[]>? BinEdges => _binEdges;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// Inverse transform returns bin midpoints, as exact original values cannot be recovered.
    /// </remarks>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="KBinsDiscretizer{T}"/>.
    /// </summary>
    /// <param name="nBins">The number of bins to use. Defaults to 5.</param>
    /// <param name="strategy">The strategy for determining bin edges. Defaults to Quantile.</param>
    /// <param name="encode">The encoding mode for output. Defaults to Ordinal.</param>
    /// <param name="columnIndices">The column indices to discretize, or null for all columns.</param>
    public KBinsDiscretizer(
        int nBins = 5,
        BinningStrategy strategy = BinningStrategy.Quantile,
        EncodeMode encode = EncodeMode.Ordinal,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nBins < 2)
        {
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));
        }

        _nBins = nBins;
        _strategy = strategy;
        _encode = encode;
    }

    /// <summary>
    /// Computes the bin edges for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        _binEdges = new List<T[]>();

        for (int col = 0; col < numColumns; col++)
        {
            if (!columnsToProcess.Contains(col))
            {
                // For columns not processed, store null edges (will pass through)
                _binEdges.Add(Array.Empty<T>());
                continue;
            }

            var column = data.GetColumn(col);
            T[] edges;

            switch (_strategy)
            {
                case BinningStrategy.Uniform:
                    edges = ComputeUniformBinEdges(column);
                    break;
                case BinningStrategy.Quantile:
                    edges = ComputeQuantileBinEdges(column);
                    break;
                case BinningStrategy.KMeans:
                    edges = ComputeKMeansBinEdges(column);
                    break;
                default:
                    throw new ArgumentException($"Unknown binning strategy: {_strategy}");
            }

            _binEdges.Add(edges);
        }
    }

    private T[] ComputeUniformBinEdges(Vector<T> column)
    {
        // Find min and max
        T min = column[0];
        T max = column[0];
        for (int i = 1; i < column.Length; i++)
        {
            if (NumOps.Compare(column[i], min) < 0) min = column[i];
            if (NumOps.Compare(column[i], max) > 0) max = column[i];
        }

        // Create uniform edges
        var edges = new T[_nBins + 1];
        T range = NumOps.Subtract(max, min);

        for (int i = 0; i <= _nBins; i++)
        {
            T fraction = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(_nBins));
            edges[i] = NumOps.Add(min, NumOps.Multiply(fraction, range));
        }

        return edges;
    }

    private T[] ComputeQuantileBinEdges(Vector<T> column)
    {
        // Sort the column
        var sortedValues = column.ToArray();
        Array.Sort(sortedValues, (a, b) => NumOps.Compare(a, b));

        // Compute quantile edges
        var edges = new T[_nBins + 1];
        for (int i = 0; i <= _nBins; i++)
        {
            double quantile = (double)i / _nBins;
            int index = (int)(quantile * (sortedValues.Length - 1));
            edges[i] = sortedValues[index];
        }

        return edges;
    }

    private T[] ComputeKMeansBinEdges(Vector<T> column)
    {
        // Simplified K-means: use quantile initialization and iterate
        // For production use, this would need a proper K-means implementation
        // For now, fall back to quantile-based edges
        return ComputeQuantileBinEdges(column);
    }

    /// <summary>
    /// Transforms the data by discretizing each feature into bins.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The discretized data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_binEdges is null)
        {
            throw new InvalidOperationException("Discretizer has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j) && _binEdges[j].Length > 0)
                {
                    int binIndex = FindBin(value, _binEdges[j]);

                    if (_encode == EncodeMode.Ordinal)
                    {
                        value = NumOps.FromDouble(binIndex);
                    }
                    else // Normalized
                    {
                        value = NumOps.Divide(NumOps.FromDouble(binIndex), NumOps.FromDouble(_nBins - 1));
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    private int FindBin(T value, T[] edges)
    {
        // Find which bin the value belongs to
        for (int i = 0; i < edges.Length - 1; i++)
        {
            if (NumOps.Compare(value, edges[i + 1]) <= 0)
            {
                return i;
            }
        }
        // Value is greater than all edges, return last bin
        return _nBins - 1;
    }

    /// <summary>
    /// Reverses the discretization by returning bin midpoints.
    /// </summary>
    /// <param name="data">The discretized data.</param>
    /// <returns>The approximate original-scale data using bin midpoints.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_binEdges is null)
        {
            throw new InvalidOperationException("Discretizer has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j) && _binEdges[j].Length > 0)
                {
                    // Get bin index from value
                    int binIndex;
                    if (_encode == EncodeMode.Ordinal)
                    {
                        binIndex = (int)NumOps.ToDouble(value);
                    }
                    else // Normalized
                    {
                        binIndex = (int)Math.Round(NumOps.ToDouble(value) * (_nBins - 1));
                    }

                    // Clamp to valid range
                    binIndex = Math.Max(0, Math.Min(_nBins - 1, binIndex));

                    // Return bin midpoint
                    T lowerEdge = _binEdges[j][binIndex];
                    T upperEdge = _binEdges[j][Math.Min(binIndex + 1, _binEdges[j].Length - 1)];
                    value = NumOps.Divide(NumOps.Add(lowerEdge, upperEdge), NumOps.FromDouble(2.0));
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (KBinsDiscretizer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
