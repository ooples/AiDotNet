using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using M-estimate regularization.
/// </summary>
/// <remarks>
/// <para>
/// MEstimateEncoder applies M-estimate smoothing to target encoding, which adds
/// a regularization parameter 'm' that controls shrinkage toward the global mean.
/// </para>
/// <para>
/// The formula: encoded = (n * category_mean + m * global_mean) / (n + m)
/// where n is the count of samples in the category and m is the smoothing parameter.
/// </para>
/// <para><b>For Beginners:</b> M-estimate is like adding 'm' fake samples:
/// - Each fake sample has the global mean as its target
/// - Categories with few samples get pulled toward the global mean
/// - Categories with many samples stay close to their actual mean
/// - Higher m = more smoothing toward global mean
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class MEstimateEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _m;
    private readonly MEstimateHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Dictionary<double, double>>? _encodingMap;
    private double _globalMean;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the smoothing parameter m.
    /// </summary>
    public double M => _m;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public MEstimateHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the global target mean.
    /// </summary>
    public double GlobalMean => _globalMean;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="MEstimateEncoder{T}"/>.
    /// </summary>
    /// <param name="m">Smoothing parameter (number of virtual samples). Defaults to 1.0.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseGlobalMean.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public MEstimateEncoder(
        double m = 1.0,
        MEstimateHandleUnknown handleUnknown = MEstimateHandleUnknown.UseGlobalMean,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (m < 0)
        {
            throw new ArgumentException("Smoothing parameter m must be non-negative.", nameof(m));
        }

        _m = m;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MEstimateEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder by computing M-estimate encodings.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Compute global mean
        int n = target.Length;
        double targetSum = 0;
        for (int i = 0; i < n; i++)
        {
            targetSum += NumOps.ToDouble(target[i]);
        }
        _globalMean = targetSum / n;

        _encodingMap = new Dictionary<int, Dictionary<double, double>>();

        foreach (int col in columnsToProcess)
        {
            var categoryStats = new Dictionary<double, (double Sum, int Count)>();

            // Collect statistics per category
            for (int i = 0; i < n; i++)
            {
                double categoryValue = NumOps.ToDouble(data[i, col]);
                double targetValue = NumOps.ToDouble(target[i]);

                if (!categoryStats.TryGetValue(categoryValue, out var stats))
                {
                    stats = (0, 0);
                }

                categoryStats[categoryValue] = (stats.Sum + targetValue, stats.Count + 1);
            }

            // Compute M-estimate encoding for each category
            var colEncoding = new Dictionary<double, double>();

            foreach (var kvp in categoryStats)
            {
                double categoryValue = kvp.Key;
                var (sum, count) = kvp.Value;

                // M-estimate formula: (n * mean + m * prior) / (n + m)
                // which simplifies to: (sum + m * prior) / (n + m)
                double encodedValue = (sum + _m * _globalMean) / (count + _m);
                colEncoding[categoryValue] = encodedValue;
            }

            _encodingMap[col] = colEncoding;
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data using fitted encodings.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_encodingMap is null)
        {
            throw new InvalidOperationException("MEstimateEncoder has not been fitted.");
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
                if (!processSet.Contains(j))
                {
                    result[i, j] = data[i, j];
                    continue;
                }

                double categoryValue = NumOps.ToDouble(data[i, j]);

                if (_encodingMap.TryGetValue(j, out var colEncoding) &&
                    colEncoding.TryGetValue(categoryValue, out double encodedValue))
                {
                    result[i, j] = NumOps.FromDouble(encodedValue);
                }
                else
                {
                    switch (_handleUnknown)
                    {
                        case MEstimateHandleUnknown.UseGlobalMean:
                            result[i, j] = NumOps.FromDouble(_globalMean);
                            break;
                        case MEstimateHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {j}");
                        default:
                            result[i, j] = NumOps.FromDouble(_globalMean);
                            break;
                    }
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MEstimateEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies how to handle unknown categories in MEstimateEncoder.
/// </summary>
public enum MEstimateHandleUnknown
{
    /// <summary>
    /// Use the global target mean for unknown categories.
    /// </summary>
    UseGlobalMean,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
