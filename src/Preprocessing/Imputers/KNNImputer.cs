using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Imputers;

/// <summary>
/// Imputes missing values using K-Nearest Neighbors.
/// </summary>
/// <remarks>
/// <para>
/// KNNImputer replaces missing values with the mean (or weighted mean) of the K nearest
/// neighbors found in the training set. Each sample's missing values are imputed using
/// the values from the K most similar samples that have non-missing values for that feature.
/// </para>
/// <para><b>For Beginners:</b> This imputer fills in missing values by looking at similar data points:
/// - Finds the K most similar rows that have the value you need
/// - Uses their average to fill in the missing value
/// - "Similar" is measured using Euclidean distance on non-missing features
///
/// Example: If you're missing someone's income, KNN finds K similar people
/// (same age, education, etc.) and uses their average income.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class KNNImputer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nNeighbors;
    private readonly KNNWeights _weights;
    private readonly double _missingValue;

    // Fitted parameters
    private Matrix<T>? _fitData;

    /// <summary>
    /// Gets the number of neighbors to use for imputation.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the weighting scheme used for neighbors.
    /// </summary>
    public KNNWeights Weights => _weights;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="KNNImputer{T}"/>.
    /// </summary>
    /// <param name="nNeighbors">Number of neighbors to use. Defaults to 5.</param>
    /// <param name="weights">Weight function used in prediction. Defaults to Uniform.</param>
    /// <param name="missingValue">The value to treat as missing. Defaults to NaN.</param>
    /// <param name="columnIndices">The column indices to impute, or null for all columns.</param>
    public KNNImputer(
        int nNeighbors = 5,
        KNNWeights weights = KNNWeights.Uniform,
        double missingValue = double.NaN,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nNeighbors = nNeighbors;
        _weights = weights;
        _missingValue = missingValue;
    }

    /// <summary>
    /// Stores the training data for neighbor lookup.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        // Store a copy of the training data for neighbor lookup
        _fitData = new Matrix<T>(data.Rows, data.Columns);
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                _fitData[i, j] = data[i, j];
            }
        }
    }

    /// <summary>
    /// Imputes missing values using K-nearest neighbors.
    /// </summary>
    /// <param name="data">The data to impute.</param>
    /// <returns>The data with missing values imputed.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_fitData is null)
        {
            throw new InvalidOperationException("Imputer has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        // Copy all data first
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                result[i, j] = data[i, j];
            }
        }

        // For each row, find missing values and impute
        for (int i = 0; i < numRows; i++)
        {
            var row = data.GetRow(i);
            var missingCols = new List<int>();

            // Find missing columns in this row
            for (int j = 0; j < numColumns; j++)
            {
                if (processSet.Contains(j) && IsMissing(row[j]))
                {
                    missingCols.Add(j);
                }
            }

            if (missingCols.Count == 0) continue;

            // Find K nearest neighbors based on non-missing features
            var (neighborIndices, neighborDistances) = FindKNearestNeighbors(row, missingCols);

            // Impute each missing column
            foreach (int col in missingCols)
            {
                result[i, col] = ImputeValue(col, neighborIndices, neighborDistances);
            }
        }

        return new Matrix<T>(result);
    }

    private bool IsMissing(T value)
    {
        double val = NumOps.ToDouble(value);
        if (double.IsNaN(_missingValue))
        {
            return double.IsNaN(val);
        }
        return Math.Abs(val - _missingValue) < 1e-10;
    }

    private (int[] Indices, double[] Distances) FindKNearestNeighbors(Vector<T> queryRow, List<int> missingCols)
    {
        if (_fitData is null)
        {
            throw new InvalidOperationException("Fit data is not available.");
        }

        var distances = new List<(int Index, double Distance)>();
        var missingSet = new HashSet<int>(missingCols);

        for (int i = 0; i < _fitData.Rows; i++)
        {
            double dist = CalculateDistance(queryRow, _fitData.GetRow(i), missingSet);
            if (!double.IsNaN(dist) && !double.IsInfinity(dist))
            {
                distances.Add((i, dist));
            }
        }

        // Sort by distance and take K nearest
        var nearest = distances.OrderBy(d => d.Distance).Take(_nNeighbors).ToArray();

        return (
            nearest.Select(n => n.Index).ToArray(),
            nearest.Select(n => n.Distance).ToArray()
        );
    }

    private double CalculateDistance(Vector<T> a, Vector<T> b, HashSet<int> ignoreCols)
    {
        double sumSq = 0;
        int count = 0;

        for (int j = 0; j < a.Length; j++)
        {
            if (ignoreCols.Contains(j)) continue;

            double va = NumOps.ToDouble(a[j]);
            double vb = NumOps.ToDouble(b[j]);

            if (double.IsNaN(va) || double.IsNaN(vb)) continue;

            double diff = va - vb;
            sumSq += diff * diff;
            count++;
        }

        if (count == 0) return double.PositiveInfinity;

        // Normalize by number of features used
        return Math.Sqrt(sumSq / count) * Math.Sqrt(a.Length);
    }

    private T ImputeValue(int col, int[] neighborIndices, double[] neighborDistances)
    {
        if (_fitData is null)
        {
            throw new InvalidOperationException("Fit data is not available.");
        }

        var validValues = new List<double>();
        var validDistances = new List<double>();

        for (int i = 0; i < neighborIndices.Length; i++)
        {
            int neighborIdx = neighborIndices[i];
            T neighborValue = _fitData[neighborIdx, col];

            if (!IsMissing(neighborValue))
            {
                validValues.Add(NumOps.ToDouble(neighborValue));
                validDistances.Add(neighborDistances[i]);
            }
        }

        if (validValues.Count == 0)
        {
            // Fallback to column mean if no valid neighbors
            return CalculateColumnMean(col);
        }

        double result;
        if (_weights == KNNWeights.Distance)
        {
            // Distance-weighted average
            double weightSum = 0;
            double valueSum = 0;

            for (int i = 0; i < validValues.Count; i++)
            {
                double weight = validDistances[i] < 1e-10 ? 1e10 : 1.0 / validDistances[i];
                weightSum += weight;
                valueSum += weight * validValues[i];
            }

            result = valueSum / weightSum;
        }
        else
        {
            // Uniform weights - simple average
            result = validValues.Average();
        }

        return NumOps.FromDouble(result);
    }

    private T CalculateColumnMean(int col)
    {
        if (_fitData is null)
        {
            return NumOps.Zero;
        }

        double sum = 0;
        int count = 0;

        for (int i = 0; i < _fitData.Rows; i++)
        {
            T value = _fitData[i, col];
            if (!IsMissing(value))
            {
                sum += NumOps.ToDouble(value);
                count++;
            }
        }

        return count > 0 ? NumOps.FromDouble(sum / count) : NumOps.Zero;
    }

    /// <summary>
    /// Inverse transformation is not supported for KNN imputation.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KNNImputer does not support inverse transformation.");
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
/// Weight function used in KNN prediction.
/// </summary>
public enum KNNWeights
{
    /// <summary>
    /// All neighbors are weighted equally.
    /// </summary>
    Uniform,

    /// <summary>
    /// Neighbors are weighted by the inverse of their distance.
    /// </summary>
    Distance
}
