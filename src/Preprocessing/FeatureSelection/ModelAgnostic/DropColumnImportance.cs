using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;

/// <summary>
/// Drop-Column Importance for model-agnostic feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Drop-Column Importance measures feature importance by removing each feature
/// entirely and measuring how much model performance degrades. This is more
/// thorough than permutation but requires retraining the model for each feature.
/// </para>
/// <para><b>For Beginners:</b> This method is like permutation importance but more
/// definitive. Instead of scrambling a feature, it completely removes it and
/// retrains the model. If the model gets much worse without a feature, that
/// feature was truly important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DropColumnImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, T>? _scorer;

    private Vector<T>? _importanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? ImportanceScores => _importanceScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DropColumnImportance(
        int nFeaturesToSelect = 10,
        Func<Matrix<T>, Vector<T>, T>? scorer = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _scorer = scorer;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DropColumnImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        var scorer = _scorer ?? DefaultScorer;

        // Compute baseline score with all features
        var baselineScore = PreprocessingTensorOps.Filled(new[] { 1 }, scorer(data, target));

        // For each feature, drop it, refit and measure performance
        var drops = new Tensor<T>[p];
        for (int dropIdx = 0; dropIdx < p; dropIdx++)
        {
            int dropped = dropIdx;
            var kept = Enumerable.Range(0, p).Where(j => j != dropped).ToArray();
            var reducedData = PreprocessingTensorOps.SelectColumns(data, kept);

            // Importance = drop in score when feature is removed
            drops[dropIdx] = engine.TensorSubtract(
                baselineScore, PreprocessingTensorOps.Filled(new[] { 1 }, scorer(reducedData, target)));
        }

        var importanceScores = p == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(drops, 0);
        _importanceScores = importanceScores.ToVector();

        // Select top features by importance
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importanceScores, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// Default score: the training R^2 of a least-squares linear model with intercept (ridge 1e-6) refitted on
    /// the given columns; 0 without columns or for a constant target.
    /// </summary>
    internal static T DefaultScorer(Matrix<T> data, Vector<T> target)
    {
        if (data.Columns == 0)
            return MathHelper.GetNumericOperations<T>().Zero;

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        return PopulationSearchOps.Value(SubsetRegressionOps.LinearRSquared(x, y, SubsetRegressionOps.FitLinear(x, y)));
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DropColumnImportance has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DropColumnImportance does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DropColumnImportance has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
