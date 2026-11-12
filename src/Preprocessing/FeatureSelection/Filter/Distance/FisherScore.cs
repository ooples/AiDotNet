using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Distance;

/// <summary>
/// Fisher Score feature selection for classification.
/// </summary>
/// <remarks>
/// <para>
/// Fisher Score measures the discriminative power of each feature by computing
/// the ratio of between-class variance to within-class variance.
/// </para>
/// <para>
/// For feature f, the Fisher Score is:
/// <code>
/// FisherScore(f) = Sum_c( n_c * (μ_c - μ)² ) / Sum_c( n_c * σ_c² )
/// </code>
/// where μ_c is the mean of feature f in class c, μ is the global mean,
/// σ_c² is the variance in class c, and n_c is the class size.
/// </para>
/// <para><b>For Beginners:</b> Fisher Score answers the question:
/// "How well does this feature separate different classes?"
///
/// A high Fisher Score means:
/// - Classes have very different average values for this feature (high between-class variance)
/// - Within each class, values are similar (low within-class variance)
///
/// Example: If predicting "is_spam", a feature like "number of exclamation marks"
/// might have a high Fisher Score because spam emails have many (!!!!) while
/// normal emails have few.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FisherScore<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    // Fitted parameters
    private Vector<T>? _scores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the computed Fisher Scores for each feature.
    /// </summary>
    public Vector<T>? Scores => _scores;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="FisherScore{T}"/>.
    /// </summary>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 10.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public FisherScore(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FisherScore requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits Fisher Score by computing discriminative power for each feature.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values (class labels).</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var y = Tensor<T>.FromVector(target);

        // Group by class (distinct target values)
        var classLabels = PreprocessingTensorOps.SortedUnique(y);
        var classIndex = PreprocessingTensorOps.LookupSorted(classLabels, y, out _);

        // Fisher Score = between-class / within-class sum of squares; when within is not above 1e-10 the score is
        // T.MaxValue if between exceeds 1e-10 (classes perfectly separated) and 0 otherwise
        var betweenClassSum = FeatureSelectionTensorOps.ClassScatter(Tensor<T>.FromMatrix(data), classIndex,
            classLabels.Length, out var withinClassSum);
        var shape = new[] { p };
        T epsilon = NumOps.FromDouble(1e-10);
        var usable = engine.TensorGreaterThan(withinClassSum, epsilon);
        var degenerate = engine.TensorWhere(engine.TensorGreaterThan(betweenClassSum, epsilon),
            PreprocessingTensorOps.Filled(shape, NumOps.MaxValue), new Tensor<T>(shape));
        var scores = engine.TensorWhere(usable, engine.TensorDivide(betweenClassSum,
            engine.TensorWhere(usable, withinClassSum, PreprocessingTensorOps.Filled(shape, NumOps.One))), degenerate);
        _scores = scores.ToVector();

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, nToSelect);

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
    /// Transforms the data by selecting Fisher-scored features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("FisherScore has not been fitted.");
        }

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FisherScore does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("FisherScore has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
        {
            mask[idx] = true;
        }

        return mask;
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
