using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Imputers;

/// <summary>
/// Creates binary indicator features for missing values.
/// </summary>
/// <remarks>
/// <para>
/// MissingIndicator transforms a dataset by adding binary columns that indicate
/// where values were missing. This is useful when the fact that a value is missing
/// is itself informative for the model.
/// </para>
/// <para><b>For Beginners:</b> Sometimes knowing that data is missing is important:
/// - A missing income might mean someone declined to answer (high income?)
/// - A missing medical test might mean the doctor didn't think it was necessary
///
/// This transformer adds new columns (one per feature) with 1 where data was missing
/// and 0 where it was present.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class MissingIndicator<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly MissingIndicatorFeatures _features;
    private readonly double _missingValue;

    // Fitted parameters
    private int[]? _featuresWithMissing;
    private int _nInputFeatures;

    /// <summary>
    /// Gets which features to create indicators for.
    /// </summary>
    public MissingIndicatorFeatures Features => _features;

    /// <summary>
    /// Gets the indices of features that had missing values during fit.
    /// </summary>
    public int[]? FeaturesWithMissing => _featuresWithMissing;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="MissingIndicator{T}"/>.
    /// </summary>
    /// <param name="features">Which features to create indicators for. Defaults to MissingOnly.</param>
    /// <param name="missingValue">The value to treat as missing. Defaults to NaN.</param>
    /// <param name="columnIndices">The column indices to check, or null for all columns.</param>
    public MissingIndicator(
        MissingIndicatorFeatures features = MissingIndicatorFeatures.MissingOnly,
        double missingValue = double.NaN,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _features = features;
        _missingValue = missingValue;
    }

    /// <summary>
    /// Identifies which features have missing values.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        if (_features == MissingIndicatorFeatures.All)
        {
            _featuresWithMissing = columnsToProcess;
        }
        else
        {
            // Find columns that actually have missing values
            var withMissing = new List<int>();

            foreach (int col in columnsToProcess)
            {
                for (int i = 0; i < data.Rows; i++)
                {
                    if (IsMissing(data[i, col]))
                    {
                        withMissing.Add(col);
                        break;
                    }
                }
            }

            _featuresWithMissing = withMissing.ToArray();
        }
    }

    /// <summary>
    /// Creates binary indicator features for missing values.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>Binary indicators for missing values.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_featuresWithMissing is null)
        {
            throw new InvalidOperationException("Indicator has not been fitted.");
        }

        int numRows = data.Rows;
        int numOutputCols = _featuresWithMissing.Length;
        var result = new T[numRows, numOutputCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numOutputCols; j++)
            {
                int sourceCol = _featuresWithMissing[j];
                result[i, j] = IsMissing(data[i, sourceCol]) ? NumOps.One : NumOps.Zero;
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

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MissingIndicator does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_featuresWithMissing is null)
        {
            return Array.Empty<string>();
        }

        var names = new string[_featuresWithMissing.Length];
        for (int i = 0; i < _featuresWithMissing.Length; i++)
        {
            int col = _featuresWithMissing[i];
            string baseName = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";
            names[i] = $"{baseName}_missing";
        }

        return names;
    }
}

/// <summary>
/// Specifies which features to create missing indicators for.
/// </summary>
public enum MissingIndicatorFeatures
{
    /// <summary>
    /// Create indicators only for features that have missing values in the training data.
    /// </summary>
    MissingOnly,

    /// <summary>
    /// Create indicators for all specified features.
    /// </summary>
    All
}
