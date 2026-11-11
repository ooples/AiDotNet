using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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
        var values = Tensor<T>.FromVector(column);
        T min = Engine.TensorMinValue(values);
        T max = Engine.TensorMaxValue(values);

        // Create uniform edges: min + (i / nBins) * (max - min)
        T range = NumOps.Subtract(max, min);
        var fraction = Engine.TensorDivideScalar(PreprocessingTensorOps.Range<T>(_nBins + 1), NumOps.FromDouble(_nBins));
        return Engine.TensorAddScalar(Engine.TensorMultiplyScalar(fraction, range), min).AsSpan().ToArray();
    }

    private T[] ComputeQuantileBinEdges(Vector<T> column)
    {
        // Sort the column
        var sortedValues = Engine.TensorSort(Tensor<T>.FromVector(column)).Item1;

        // Compute quantile edges
        var index = new int[_nBins + 1];
        for (int i = 0; i <= _nBins; i++)
        {
            double quantile = (double)i / _nBins;
            index[i] = (int)(quantile * (sortedValues.Length - 1));
        }

        return Engine.TensorIndexSelect(sortedValues, new Tensor<int>(index, new[] { _nBins + 1 }), 0).AsSpan().ToArray();
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

        var engine = Engine;
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var columns = new List<Tensor<T>>(numColumns);

        for (int j = 0; j < numColumns; j++)
        {
            var column = PreprocessingTensorOps.Column(data, j);
            if (processSet.Contains(j) && _binEdges[j].Length > 0)
            {
                var binIndex = FindBins(column, _binEdges[j]).Cast<T>();
                column = _encode == EncodeMode.Ordinal
                    ? binIndex
                    : engine.TensorDivideScalar(binIndex, NumOps.FromDouble(_nBins - 1)); // Normalized
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// The bin of each value: the first bin whose upper edge is at least the value, or the last bin for values
    /// above every edge.
    /// </summary>
    private Tensor<int> FindBins(Tensor<T> values, T[] edges)
    {
        if (values.Length == 0)
            return new Tensor<int>(new[] { 0 });

        int upperEdges = edges.Length - 1;
        var upper = new Tensor<T>(edges.Skip(1).ToArray(), new[] { upperEdges });
        var bins = Engine.TensorSearchSorted(upper, values, false).AsSpan().ToArray();
        for (int i = 0; i < bins.Length; i++)
            bins[i] = Math.Min(bins[i], _nBins - 1);
        return new Tensor<int>(bins, new[] { bins.Length });
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

        var engine = Engine;
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var columns = new List<Tensor<T>>(numColumns);
        T bins = NumOps.FromDouble(_nBins - 1);

        for (int j = 0; j < numColumns; j++)
        {
            var column = PreprocessingTensorOps.Column(data, j);
            var edgeValues = _binEdges[j];
            if (processSet.Contains(j) && edgeValues.Length > 0 && numRows > 0)
            {
                // Get bin index from value (truncated for ordinal codes, rounded for normalized ones)
                var binIndex = _encode == EncodeMode.Ordinal
                    ? engine.TensorTrunc(column)
                    : engine.TensorRound(engine.TensorMultiplyScalar(column, bins));

                // Clamp to valid range
                var lower = engine.TensorClamp(binIndex, NumOps.Zero, bins);
                var lowerIndex = lower.Cast<int>();
                var upperIndex = engine.TensorClamp(engine.TensorAddScalar(lower, NumOps.One), NumOps.Zero,
                    NumOps.FromDouble(edgeValues.Length - 1)).Cast<int>();

                // Return bin midpoint
                var edges = new Tensor<T>(edgeValues, new[] { edgeValues.Length });
                column = engine.TensorDivideScalar(
                    engine.TensorAdd(engine.TensorIndexSelect(edges, lowerIndex, 0), engine.TensorIndexSelect(edges, upperIndex, 0)),
                    NumOps.FromDouble(2.0));
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
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
