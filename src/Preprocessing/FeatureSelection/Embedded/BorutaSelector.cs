using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Boruta feature selection using shadow features and random forest importance.
/// </summary>
/// <remarks>
/// <para>
/// Boruta creates "shadow" features by shuffling original features. A random forest
/// is trained, and features that consistently outperform the best shadow feature
/// are confirmed as important. Features that never beat shadows are rejected.
/// </para>
/// <para><b>For Beginners:</b> Imagine testing each feature against a "random noise"
/// version of itself. If a real feature is important, it should beat its noisy twin
/// repeatedly. Boruta does this systematically, keeping only features that clearly
/// outperform random chance.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BorutaSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _nIterations;
    private readonly int _nTrees;
    private readonly double _percentile;
    private readonly int? _randomState;

    private Vector<T>? _featureImportances;
    private bool[]? _confirmedFeatures;
    private bool[]? _tentativeFeatures;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NIterations => _nIterations;
    public Vector<T>? FeatureImportances => _featureImportances;
    public bool[]? ConfirmedFeatures => _confirmedFeatures;
    public bool[]? TentativeFeatures => _tentativeFeatures;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BorutaSelector(
        int nIterations = 100,
        int nTrees = 50,
        double percentile = 100.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _nIterations = nIterations;
        _nTrees = nTrees;
        _percentile = percentile;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BorutaSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
            columns[j] = PreprocessingTensorOps.ColumnVector(X, j);

        // Class labels (rounded target values) for the Gini stumps
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);

        // Track hits (times feature beats max shadow)
        var hits = new int[p];
        var importanceSums = new Tensor<T>(new[] { p });
        T zero = NumOps.Zero;

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Create shadow features by shuffling, one permutation per feature in order
            var shadows = new Tensor<T>[p];
            for (int j = 0; j < p; j++)
            {
                var shuffled = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                shadows[j] = engine.Reshape(
                    engine.TensorIndexSelect(columns[j], new Tensor<int>(shuffled, new[] { n }), 0), new[] { n, 1 });
            }

            var extendedData = engine.TensorConcatenate(new[] { X, engine.TensorConcatenate(shadows, 1) }, 1);

            // Train simplified random forest and get importance
            var importances = ComputeRandomForestImportance(extendedData, classIndex, classes.Length, n, p * 2, random);
            var original = engine.TensorSlice(importances, new[] { 0 }, new[] { p });

            // Find max shadow importance (at least 0)
            T maxShadow = PreprocessingTensorOps.ScalarMax(zero,
                engine.TensorMaxValue(engine.TensorSlice(importances, new[] { p }, new[] { p })));

            // Count hits
            importanceSums = engine.TensorAdd(importanceSums, original);
            foreach (int j in PreprocessingTensorOps.MaskPositions(engine.TensorGreaterThan(original, maxShadow)))
                hits[j]++;
        }

        // Normalize importance
        var featureImportances = engine.TensorDivideScalar(importanceSums, NumOps.FromDouble(_nIterations));
        _featureImportances = featureImportances.ToVector();

        // Statistical test for confirmation (simplified binomial test)
        _confirmedFeatures = new bool[p];
        _tentativeFeatures = new bool[p];

        double expectedHits = _nIterations * 0.5; // Under null hypothesis
        double stdDev = Math.Sqrt(_nIterations * 0.5 * 0.5);

        for (int j = 0; j < p; j++)
        {
            double zScore = (hits[j] - expectedHits) / stdDev;

            if (zScore > 2.0) // Confirmed
                _confirmedFeatures[j] = true;
            else if (zScore > 0) // Tentative
                _tentativeFeatures[j] = true;
        }

        // Select confirmed features (plus tentative if needed)
        var selected = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_confirmedFeatures[j])
                selected.Add(j);
        }

        // If no confirmed, add tentative
        if (selected.Count == 0)
        {
            for (int j = 0; j < p; j++)
            {
                if (_tentativeFeatures[j])
                    selected.Add(j);
            }
        }

        // If still none, select top by importance
        if (selected.Count == 0)
        {
            selected = PreprocessingTensorOps.TopKIndicesSorted(featureImportances, Math.Min(10, p)).ToList();
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private Tensor<T> ComputeRandomForestImportance(
        Tensor<T> data, int[] classIndex, int classCount, int n, int p, Random random)
    {
        var importances = new Tensor<T>(new[] { p });

        for (int t = 0; t < _nTrees; t++)
        {
            // Bootstrap sample
            var sampleIndices = new int[n];
            var sampleClasses = new int[n];
            for (int i = 0; i < n; i++)
            {
                sampleIndices[i] = random.Next(n);
                sampleClasses[i] = classIndex[sampleIndices[i]];
            }

            // Best Gini-reduction decision stump on every feature
            var sample = Engine.TensorIndexSelect(data, new Tensor<int>(sampleIndices, new[] { n }), 0);
            importances = Engine.TensorAdd(importances,
                FeatureSelectionTensorOps.BestGiniSplitReduction(sample, sampleClasses, classCount));
        }

        // Normalize
        return Engine.TensorDivideScalar(importances, NumOps.FromDouble(_nTrees));
    }
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BorutaSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BorutaSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BorutaSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
