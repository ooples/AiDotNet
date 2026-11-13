using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Imbalanced;

/// <summary>
/// SMOTE-aware Feature Selection for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para>
/// This feature selector is designed for imbalanced datasets where one class is
/// significantly underrepresented. It evaluates features after synthetic minority
/// oversampling to ensure selected features work well on balanced data.
/// </para>
/// <para><b>For Beginners:</b> When you have few examples of one class (like rare diseases),
/// regular feature selection might ignore patterns specific to that class. This method
/// creates synthetic examples of the minority class first, then selects features that
/// distinguish both classes well.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SMOTEFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _samplingRatio;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double SamplingRatio => _samplingRatio;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SMOTEFeatureSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        double samplingRatio = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        if (samplingRatio <= 0 || samplingRatio > 2)
            throw new ArgumentException("Sampling ratio must be between 0 and 2.", nameof(samplingRatio));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _samplingRatio = samplingRatio;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SMOTEFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Split data by class
        var x = Tensor<T>.FromMatrix(data);
        var isClass0 = engine.TensorLessThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5));
        var class0Indices = PreprocessingTensorOps.MaskPositions(isClass0);
        var class1Indices = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.OneMinus(isClass0));

        // Determine minority class
        var (minorityIndices, majorityIndices) = class0Indices.Length < class1Indices.Length
            ? (class0Indices, class1Indices)
            : (class1Indices, class0Indices);

        var minority = Rows(x, minorityIndices);
        var majority = Rows(x, majorityIndices);

        // Generate synthetic samples using SMOTE (Chawla et al. 2002); each needs at least one minority neighbour
        int nSynthetic = (int)((majorityIndices.Length - minorityIndices.Length) * _samplingRatio);
        Tensor<T>? synthetic = null;
        if (nSynthetic > 0 && minorityIndices.Length >= _nNeighbors && minorityIndices.Length > 1)
        {
            var neighbors = MinorityNeighbors(minority, _nNeighbors);
            var baseRows = new int[nSynthetic];
            var neighborRows = new int[nSynthetic];
            var lambdas = new Tensor<T>[nSynthetic];
            for (int s = 0; s < nSynthetic; s++)
            {
                // Pick random minority sample, then a random one of its k nearest minority neighbours
                baseRows[s] = random.Next(minorityIndices.Length);
                var own = neighbors[baseRows[s]];
                neighborRows[s] = own[random.Next(own.Length)];
                lambdas[s] = PopulationSearchOps.Uniform<T>(random, 1);
            }

            synthetic = Interpolate(minority, baseRows, neighborRows, engine.TensorConcatenate(lambdas, 0));
        }

        // Compute feature scores on balanced data
        var minorityAll = synthetic is null ? minority : engine.TensorConcatenate(new[] { minority, synthetic }, 0);
        var scores = majorityIndices.Length == 0 || minorityAll.Shape[0] == 0
            ? new Tensor<T>(new[] { p })
            : FisherScores(majority, minorityAll);
        _featureScores = scores.ToVector();

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    private static Tensor<T> Rows(Tensor<T> x, int[] rows) =>
        rows.Length == 0
            ? new Tensor<T>(new[] { 0, x.Shape[1] })
            : AiDotNetEngine.Current.TensorIndexSelect(x, new Tensor<int>((int[])rows.Clone(), new[] { rows.Length }), 0);

    /// <summary>
    /// For every minority row, the positions (within <paramref name="minority"/>) of its
    /// <paramref name="k"/> nearest other minority rows by Euclidean distance, ties to the lower position.
    /// </summary>
    internal static int[][] MinorityNeighbors(Tensor<T> minority, int k)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = minority.Shape[0];
        int count = Math.Min(k, m - 1);

        // The row itself is pushed past every other row so it is never its own neighbour. The neighbour order does
        // not depend on the scale, so the rows are divided by their largest magnitude first: the distance expansion
        // squares the values, which passes 65504 in Half past 256 and left every distance NaN.
        var eye = engine.TensorEye<T>(m);
        var largest = engine.ReduceMax(engine.TensorAbs(minority), new[] { 0, 1 }, true);
        var scale = engine.TensorWhere(
            engine.TensorGreaterThan(largest, ops.Zero), largest, PreprocessingTensorOps.Filled(largest.Shape.ToArray(), ops.One));
        var unit = engine.TensorDivide(minority, engine.TensorBroadcastTo(scale, minority.Shape.ToArray()));
        var distances = engine.TensorWhere(
            eye, PreprocessingTensorOps.Filled(new[] { m, m }, ops.MaxValue), FeatureSelectionTensorOps.PairwiseDistances(unit));
        var neighbors = new int[m][];
        for (int i = 0; i < m; i++)
            neighbors[i] = PreprocessingTensorOps.StableOrder(PopulationSearchOps.Row(distances, i), descending: false)
                .Take(count)
                .ToArray();
        return neighbors;
    }

    /// <summary>
    /// SMOTE samples base + lambda (neighbour - base) for row pairs (<paramref name="baseRows"/>[s],
    /// <paramref name="neighborRows"/>[s]) of <paramref name="minority"/>, with lambda [s] in [0, 1).
    /// </summary>
    internal static Tensor<T> Interpolate(Tensor<T> minority, int[] baseRows, int[] neighborRows, Tensor<T> lambdas)
    {
        var engine = AiDotNetEngine.Current;
        int s = baseRows.Length;
        int p = minority.Shape[1];
        var bases = Rows(minority, baseRows);
        var others = Rows(minority, neighborRows);
        var step = engine.TensorBroadcastTo(engine.Reshape(lambdas, new[] { s, 1 }), new[] { s, p });
        return engine.TensorAdd(bases, engine.TensorMultiply(step, engine.TensorSubtract(others, bases)));
    }

    /// <summary>
    /// Fisher's discriminant ratio (mean0 - mean1)^2 / (var0 + var1) per feature of two classes [n0, p] and
    /// [n1, p], with population variances; 0 where the variances sum to at most 1e-10.
    /// </summary>
    private Tensor<T> FisherScores(Tensor<T> class0, Tensor<T> class1)
    {
        var engine = Engine;
        int p = class0.Shape[1];
        var mean0 = engine.ReduceMean(class0, new[] { 0 }, true);
        var mean1 = engine.ReduceMean(class1, new[] { 0 }, true);
        var var0 = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(class0, engine.TensorBroadcastTo(mean0, class0.Shape.ToArray()))), new[] { 0 }, false);
        var var1 = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(class1, engine.TensorBroadcastTo(mean1, class1.Shape.ToArray()))), new[] { 0 }, false);
        var denom = engine.TensorAdd(var0, var1);
        var valid = engine.TensorGreaterThan(denom, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var safe = engine.TensorWhere(valid, denom, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        var gap = engine.TensorSquare(engine.Reshape(engine.TensorSubtract(mean0, mean1), new[] { p }));
        return engine.TensorWhere(valid, engine.TensorDivide(gap, safe), new Tensor<T>(new[] { p }));
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
            throw new InvalidOperationException("SMOTEFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SMOTEFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SMOTEFeatureSelector has not been fitted.");

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
