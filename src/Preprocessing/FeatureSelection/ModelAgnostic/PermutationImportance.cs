using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;

/// <summary>
/// Permutation Importance for model-agnostic feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Permutation Importance measures feature importance by randomly shuffling each feature
/// and observing how much model performance degrades. Features that cause large
/// performance drops when shuffled are important.
/// </para>
/// <para><b>For Beginners:</b> This method works with any model. It asks: "What happens
/// if I scramble this feature's values?" If the model gets much worse, that feature
/// was important. If the model doesn't care, the feature wasn't useful. It's like
/// testing each ingredient by removing it from a recipe.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PermutationImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nPermutations;
    private readonly Func<Matrix<T>, Vector<T>, T>? _scorer;
    private readonly int? _randomState;

    private Vector<T>? _importanceScores;
    private Vector<T>? _importanceStd;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NPermutations => _nPermutations;
    public Vector<T>? ImportanceScores => _importanceScores;
    public Vector<T>? ImportanceStd => _importanceStd;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PermutationImportance(
        int nFeaturesToSelect = 10,
        int nPermutations = 10,
        Func<Matrix<T>, Vector<T>, T>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nPermutations < 1)
            throw new ArgumentException("Number of permutations must be at least 1.", nameof(nPermutations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nPermutations = nPermutations;
        _scorer = scorer;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PermutationImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // The default model is fitted once on the original data and then scored on every permuted copy
        Func<Matrix<T>, Vector<T>, T> scorer;
        if (_scorer is not null)
        {
            scorer = _scorer;
        }
        else
        {
            var coefficients = SubsetRegressionOps.FitLinear(x, y);
            scorer = (m, v) => PopulationSearchOps.Value(
                SubsetRegressionOps.LinearRSquared(Tensor<T>.FromMatrix(m), Tensor<T>.FromVector(v), coefficients));
        }

        // Compute baseline score
        var baselineScore = PreprocessingTensorOps.Filled(new[] { 1 }, scorer(data, target));

        var importances = new Tensor<T>[p];
        var deviations = new Tensor<T>[p];

        // For each feature, permute and measure performance drop (Breiman 2001)
        for (int j = 0; j < p; j++)
        {
            var column = engine.TensorBroadcastTo(
                engine.Reshape(FeatureSelectionTensorOps.OneHot<T>(j, p), new[] { 1, p }), new[] { n, p });
            var drops = new Tensor<T>[_nPermutations];

            for (int perm = 0; perm < _nPermutations; perm++)
            {
                // Shuffle column j
                var shuffledIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                var shuffled = engine.TensorIndexSelect(x, new Tensor<int>(shuffledIndices, new[] { n }), 0);
                var permuted = engine.TensorWhere(column, shuffled, x).ToMatrix();

                // Importance = drop in score
                var permScore = PreprocessingTensorOps.Filled(new[] { 1 }, scorer(permuted, target));
                drops[perm] = engine.TensorSubtract(baselineScore, permScore);
            }

            // Mean importance and (population) standard deviation over the permutations
            var all = engine.TensorConcatenate(drops, 0);
            var mean = engine.ReduceMean(all, new[] { 0 }, true);
            importances[j] = mean;
            deviations[j] = engine.TensorSqrt(engine.ReduceMean(
                engine.TensorSquare(engine.TensorSubtract(all, engine.TensorBroadcastTo(mean, new[] { _nPermutations }))),
                new[] { 0 }, true));
        }

        var importanceScores = engine.TensorConcatenate(importances, 0);
        _importanceScores = importanceScores.ToVector();
        _importanceStd = engine.TensorConcatenate(deviations, 0).ToVector();

        // Select top features by importance
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importanceScores, numToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("PermutationImportance has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PermutationImportance does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationImportance has not been fitted.");

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
