using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Feature selector that removes features with variance below a threshold.
/// </summary>
/// <remarks>
/// <para>
/// VarianceThreshold removes all features whose variance doesn't meet a minimum threshold.
/// Features with low variance are often not informative because they don't vary enough
/// across samples to be useful for prediction.
/// </para>
/// <para><b>For Beginners:</b> If a feature has the same value (or nearly the same value)
/// for all samples, it won't help your model distinguish between different outcomes.
/// This transformer automatically removes such features:
/// - Constant features (all same value) have variance = 0
/// - Near-constant features have very low variance
///
/// Example: A "Country" column that is "USA" for 99.9% of rows provides little information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class VarianceThreshold<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;

    // Fitted parameters
    private double[]? _variances;
    private int[]? _selectedFeatures;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the variance threshold.
    /// </summary>
    public double Threshold => _threshold;

    /// <summary>
    /// Gets the computed variances for each feature.
    /// </summary>
    public double[]? Variances => _variances;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedFeatures => _selectedFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="VarianceThreshold{T}"/>.
    /// </summary>
    /// <param name="threshold">Features with variance below this are removed. Defaults to 0.0.</param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public VarianceThreshold(
        double threshold = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0)
        {
            throw new ArgumentException("Threshold must be non-negative.", nameof(threshold));
        }

        _threshold = threshold;
    }

    /// <summary>
    /// Computes the variance of each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        _variances = new double[_nInputFeatures];
        var selectedList = new List<int>();

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (!processSet.Contains(col))
            {
                // Pass-through columns always selected
                _variances[col] = double.MaxValue;
                selectedList.Add(col);
                continue;
            }

            // Compute mean
            double sum = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                sum += NumOps.ToDouble(data[i, col]);
            }
            double mean = sum / data.Rows;

            // Compute variance
            double sumSq = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                double diff = NumOps.ToDouble(data[i, col]) - mean;
                sumSq += diff * diff;
            }
            double variance = sumSq / data.Rows;
            _variances[col] = variance;

            // Select if above threshold
            if (variance > _threshold)
            {
                selectedList.Add(col);
            }
        }

        _selectedFeatures = selectedList.ToArray();

        if (_selectedFeatures.Length == 0)
        {
            throw new InvalidOperationException(
                $"No features meet the variance threshold of {_threshold}. " +
                "Consider lowering the threshold or checking your data for constant features.");
        }
    }

    /// <summary>
    /// Removes features with variance below the threshold.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with low-variance features removed.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedFeatures is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        int numRows = data.Rows;
        int numOutputCols = _selectedFeatures.Length;
        var result = new T[numRows, numOutputCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numOutputCols; j++)
            {
                int sourceCol = _selectedFeatures[j];
                result[i, j] = data[i, sourceCol];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("VarianceThreshold does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    /// <returns>Array where true indicates the feature is selected.</returns>
    public bool[] GetSupportMask()
    {
        if (_selectedFeatures is null || _variances is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        var selectedSet = new HashSet<int>(_selectedFeatures);

        for (int i = 0; i < _nInputFeatures; i++)
        {
            mask[i] = selectedSet.Contains(i);
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedFeatures is null)
        {
            return Array.Empty<string>();
        }

        var names = new string[_selectedFeatures.Length];
        for (int i = 0; i < _selectedFeatures.Length; i++)
        {
            int col = _selectedFeatures[i];
            names[i] = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";
        }

        return names;
    }
}
