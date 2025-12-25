using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using James-Stein shrinkage estimation.
/// </summary>
/// <remarks>
/// <para>
/// JamesSteinEncoder uses Bayesian shrinkage to blend category-specific target means
/// with the global mean. Categories with more samples get weights closer to their
/// own mean, while rare categories shrink toward the global mean.
/// </para>
/// <para>
/// The shrinkage formula: encoded = (1 - B) * category_mean + B * global_mean
/// where B is the shrinkage factor based on sample size and variance.
/// </para>
/// <para><b>For Beginners:</b> This encoder balances between:
/// - Trusting category-specific averages (when we have lots of data)
/// - Falling back to the overall average (when category data is sparse)
/// - The balance is determined automatically using statistical theory
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class JamesSteinEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly JamesSteinHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Dictionary<double, double>>? _encodingMap;
    private double _globalMean;
    private int _nInputFeatures;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public JamesSteinHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the global target mean.
    /// </summary>
    public double GlobalMean => _globalMean;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="JamesSteinEncoder{T}"/>.
    /// </summary>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseGlobalMean.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public JamesSteinEncoder(
        JamesSteinHandleUnknown handleUnknown = JamesSteinHandleUnknown.UseGlobalMean,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "JamesSteinEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder by computing James-Stein shrinkage estimates.
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

        // Compute global mean and variance
        int n = target.Length;
        double targetSum = 0;
        for (int i = 0; i < n; i++)
        {
            targetSum += NumOps.ToDouble(target[i]);
        }
        _globalMean = targetSum / n;

        double globalVariance = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(target[i]) - _globalMean;
            globalVariance += diff * diff;
        }
        globalVariance /= (n - 1);

        _encodingMap = new Dictionary<int, Dictionary<double, double>>();

        foreach (int col in columnsToProcess)
        {
            var categoryStats = new Dictionary<double, (double Sum, double SumSq, int Count)>();

            // Collect statistics per category
            for (int i = 0; i < n; i++)
            {
                double categoryValue = NumOps.ToDouble(data[i, col]);
                double targetValue = NumOps.ToDouble(target[i]);

                if (!categoryStats.TryGetValue(categoryValue, out var stats))
                {
                    stats = (0, 0, 0);
                }

                categoryStats[categoryValue] = (
                    stats.Sum + targetValue,
                    stats.SumSq + targetValue * targetValue,
                    stats.Count + 1
                );
            }

            // Compute James-Stein shrinkage for each category
            var colEncoding = new Dictionary<double, double>();

            foreach (var kvp in categoryStats)
            {
                double categoryValue = kvp.Key;
                var (sum, sumSq, count) = kvp.Value;

                double categoryMean = sum / count;

                // Compute shrinkage factor B using James-Stein formula
                // B = (k - 2) * sigma^2 / (n * (mean - grand_mean)^2)
                // where k is the number of categories
                double shrinkage;

                if (count <= 2 || globalVariance < 1e-10)
                {
                    shrinkage = 1.0; // Full shrinkage to global mean
                }
                else
                {
                    double diffFromGlobal = categoryMean - _globalMean;
                    double denominator = count * diffFromGlobal * diffFromGlobal;

                    if (denominator < 1e-10)
                    {
                        shrinkage = 1.0;
                    }
                    else
                    {
                        // Simplified James-Stein shrinkage
                        shrinkage = Math.Max(0, 1 - (count - 2) * globalVariance / denominator);
                        shrinkage = Math.Min(1, shrinkage);
                    }
                }

                // Apply shrinkage: (1 - B) * category_mean + B * global_mean
                double encodedValue = (1 - shrinkage) * categoryMean + shrinkage * _globalMean;
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
            throw new InvalidOperationException("JamesSteinEncoder has not been fitted.");
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
                        case JamesSteinHandleUnknown.UseGlobalMean:
                            result[i, j] = NumOps.FromDouble(_globalMean);
                            break;
                        case JamesSteinHandleUnknown.Error:
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
        throw new NotSupportedException("JamesSteinEncoder does not support inverse transformation.");
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
/// Specifies how to handle unknown categories in JamesSteinEncoder.
/// </summary>
public enum JamesSteinHandleUnknown
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
