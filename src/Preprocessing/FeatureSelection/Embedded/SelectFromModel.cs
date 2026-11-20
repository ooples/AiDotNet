using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Selects features based on importance weights from an external model or scorer.
/// </summary>
/// <remarks>
/// <para>
/// SelectFromModel selects features based on importance scores, typically from a fitted model.
/// Features with importance above a threshold are kept.
/// </para>
/// <para>
/// The threshold can be specified as:
/// - An absolute value
/// - "mean" - the mean of feature importances
/// - "median" - the median of feature importances
/// </para>
/// <para><b>For Beginners:</b> This works with any model that produces feature importances:
/// - Random forests give importance based on how much each feature reduces error
/// - Linear models give coefficients showing feature influence
/// - Features below the threshold are removed
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectFromModel<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly Func<Matrix<T>, Vector<T>, Vector<T>>? _importanceFunc;
    private readonly Vector<T>? _precomputedImportances;
    private readonly SelectFromModelThreshold _thresholdType;
    private readonly double _thresholdValue;
    private readonly int? _maxFeatures;

    // Fitted parameters
    private Vector<T>? _featureImportances;
    private T _threshold = MathHelper.GetNumericOperations<T>().Zero;
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the feature importances used for selection.
    /// </summary>
    public Vector<T>? FeatureImportances => _featureImportances;

    /// <summary>
    /// Gets the computed threshold value.
    /// </summary>
    public T Threshold => _threshold;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance with a function to compute feature importances.
    /// </summary>
    /// <param name="importanceFunc">Function that computes feature importances given data and target.</param>
    /// <param name="thresholdType">Type of threshold to use. Defaults to Mean.</param>
    /// <param name="thresholdValue">Threshold value (used when type is Value). Defaults to 0.</param>
    /// <param name="maxFeatures">Maximum number of features to select. Null for no limit.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public SelectFromModel(
        Func<Matrix<T>, Vector<T>, Vector<T>> importanceFunc,
        SelectFromModelThreshold thresholdType = SelectFromModelThreshold.Mean,
        double thresholdValue = 0.0,
        int? maxFeatures = null,
        int[]? columnIndices = null)
        : this(RequireImportanceFunc(importanceFunc), null, thresholdType, thresholdValue, maxFeatures, columnIndices)
    {
    }

    /// <summary>
    /// Creates a new instance with precomputed feature importances.
    /// </summary>
    /// <param name="featureImportances">Precomputed feature importance scores.</param>
    /// <param name="thresholdType">Type of threshold to use. Defaults to Mean.</param>
    /// <param name="thresholdValue">Threshold value (used when type is Value). Defaults to 0.</param>
    /// <param name="maxFeatures">Maximum number of features to select. Null for no limit.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    /// <returns>A selector that fits from <paramref name="featureImportances"/> alone.</returns>
    /// <remarks>
    /// A factory rather than a second constructor: the importance-function constructor is the one public
    /// constructor with optional parameters, and a precomputed-importances overload of the same length would
    /// break RS0026/RS0027 (optional parameters on one overload, which must also have the most parameters).
    /// </remarks>
    public static SelectFromModel<T> FromImportances(
        Vector<T> featureImportances,
        SelectFromModelThreshold thresholdType = SelectFromModelThreshold.Mean,
        double thresholdValue = 0.0,
        int? maxFeatures = null,
        int[]? columnIndices = null)
        => new(null, RequireImportances(featureImportances), thresholdType, thresholdValue, maxFeatures, columnIndices);

    private SelectFromModel(
        Func<Matrix<T>, Vector<T>, Vector<T>>? importanceFunc,
        Vector<T>? precomputedImportances,
        SelectFromModelThreshold thresholdType,
        double thresholdValue,
        int? maxFeatures,
        int[]? columnIndices)
        : base(columnIndices)
    {
        _importanceFunc = importanceFunc;
        _precomputedImportances = precomputedImportances;
        _thresholdType = thresholdType;
        _thresholdValue = thresholdValue;
        _maxFeatures = maxFeatures;
    }

    private static Func<Matrix<T>, Vector<T>, Vector<T>> RequireImportanceFunc(
        Func<Matrix<T>, Vector<T>, Vector<T>> importanceFunc)
    {
        Guard.NotNull(importanceFunc);
        return importanceFunc;
    }

    private static Vector<T> RequireImportances(Vector<T> featureImportances)
    {
        Guard.NotNull(featureImportances);
        return featureImportances;
    }

    /// <summary>
    /// Fits the selector using precomputed importances only.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        if (_precomputedImportances is null)
        {
            throw new InvalidOperationException(
                "SelectFromModel requires either precomputed importances or target values. " +
                "Use Fit(Matrix<T> data, Vector<T> target) when using an importance function.");
        }

        _nInputFeatures = data.Columns;

        if (_precomputedImportances.Length != _nInputFeatures)
        {
            throw new ArgumentException(
                $"Precomputed importances length ({_precomputedImportances.Length}) must match number of features ({_nInputFeatures}).");
        }

        _featureImportances = new Vector<T>(_precomputedImportances.ToArray());
        ComputeThresholdAndSelect();
        IsFitted = true;
    }

    /// <summary>
    /// Fits the selector by computing feature importances.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;

        if (_precomputedImportances is not null)
        {
            _featureImportances = new Vector<T>(_precomputedImportances.ToArray());
        }
        else if (_importanceFunc is not null)
        {
            _featureImportances = _importanceFunc(data, target);
        }
        else
        {
            throw new InvalidOperationException("No importance function or precomputed importances available.");
        }

        if (_featureImportances.Length != _nInputFeatures)
        {
            throw new ArgumentException(
                $"Importance scores length ({_featureImportances.Length}) must match number of features ({_nInputFeatures}).");
        }

        ComputeThresholdAndSelect();
        IsFitted = true;
    }

    private void ComputeThresholdAndSelect()
    {
        if (_featureImportances is null)
        {
            throw new InvalidOperationException("Feature importances not computed.");
        }

        // Use absolute values for threshold computation
        var engine = Engine;
        var absImportances = engine.TensorAbs(Tensor<T>.FromVector(_featureImportances));

        // Compute threshold
        _threshold = _thresholdType switch
        {
            SelectFromModelThreshold.Mean => engine.TensorMean(absImportances),
            SelectFromModelThreshold.Median => ComputeMedian(absImportances),
            SelectFromModelThreshold.Value => NumOps.FromDouble(_thresholdValue),
            _ => engine.TensorMean(absImportances)
        };

        // Select features at or above threshold
        var atOrAbove = engine.TensorAdd(
            engine.TensorGreaterThan(absImportances, _threshold), engine.TensorEquals(absImportances, _threshold));
        var selectedList = PreprocessingTensorOps.MaskPositions(atOrAbove).ToList();

        // Apply max features limit (largest first, ties by index)
        if (_maxFeatures.HasValue && selectedList.Count > _maxFeatures.Value)
        {
            var selectedSet = new HashSet<int>(selectedList);
            selectedList = PreprocessingTensorOps.StableOrder(absImportances, descending: true)
                .Where(i => selectedSet.Contains(i))
                .Take(_maxFeatures.Value)
                .OrderBy(i => i)
                .ToList();
        }

        // Ensure at least one feature is selected
        if (selectedList.Count == 0)
        {
            selectedList.Add(PreprocessingTensorOps.FirstArgMax(absImportances));
        }

        _selectedIndices = selectedList.ToArray();

        // Create support mask
        _supportMask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
        {
            _supportMask[idx] = true;
        }
    }

    private T ComputeMedian(Tensor<T> values)
    {
        int n = values.Length;

        if (n == 0) return NumOps.Zero;
        return Engine.TensorSum(FeatureScoringTensorOps.ColumnMedian(Engine.Reshape(values, new[] { n, 1 })));
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
    /// Transforms the data by selecting important features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SelectFromModel has not been fitted.");
        }

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);

    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectFromModel does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("SelectFromModel has not been fitted.");
        }
        return (bool[])_supportMask.Clone();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}

/// <summary>
/// Specifies the type of threshold for SelectFromModel.
/// </summary>
public enum SelectFromModelThreshold
{
    /// <summary>
    /// Use the mean of feature importances as threshold.
    /// </summary>
    Mean,

    /// <summary>
    /// Use the median of feature importances as threshold.
    /// </summary>
    Median,

    /// <summary>
    /// Use a specified value as threshold.
    /// </summary>
    Value
}
